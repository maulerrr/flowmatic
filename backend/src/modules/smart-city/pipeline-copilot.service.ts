import { BadRequestException, Injectable, Logger } from '@nestjs/common'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { DataRow } from 'src/common/types/data.types'
import { AppConfigService } from 'src/common/config/config.service'
import { AuthContext } from 'src/modules/auth/auth-context.service'
import { QualityService } from '../quality/quality.service'
import { PrismaService } from 'src/prisma/prisma.service'
import { SmartCityService } from './smart-city.service'
import {
	CopilotChip,
	CopilotVizSpec,
	PipelineCopilotChatResponse,
	PipelineCopilotContext,
	PipelineCopilotExportRunSummary,
} from './pipeline-copilot.types'
import {
	buildVizCommandsForChips,
	parseComposerMessage,
} from './pipeline-copilot.commands'

const WINDOW_HOURS = 24

const COPILOT_SYSTEM_PROMPT = `You are Flowmatic Pipeline Insights, an assistant for smart-city realtime data pipelines.

You receive a JSON context describing ONE pipeline: ingest sources, core processing model, medallion lake stages (raw/cleaned/business), export targets, and activity in the last 24 hours.

Rules:
- Answer ONLY from the provided context. Never invent metrics, sources, or exports.
- Explain clearly for an operator: what the data is, what the pipeline is doing right now, and any issues.
- Use 2–4 short paragraphs in plain language. No markdown headings or code fences.
- Cite concrete numbers from context when helpful.
- If the pipeline is paused or sources are stopped, explain that and mention starting it from the workbench.
- If the user asks about a specific area (exports, sources, quality, model), focus on that while staying grounded in context.`

/**
 * Pipeline copilot: live context is computed at request time; chat turns are persisted
 * server-side (per org + pipeline + user). All LLM calls run on the backend only.
 */
@Injectable()
export class PipelineCopilotService {
	private readonly logger = new Logger(PipelineCopilotService.name)
	private readonly chatModel?: ChatOpenAI

	constructor(
		private readonly prisma: PrismaService,
		private readonly smartCity: SmartCityService,
		private readonly qualityService: QualityService,
		private readonly config: AppConfigService,
	) {
		if (this.config.openai.apiKey) {
			this.chatModel = new ChatOpenAI({
				modelName: this.config.openai.model,
				temperature: 0.25,
				openAIApiKey: this.config.openai.apiKey,
			})
			this.logger.log(`Pipeline copilot LLM enabled (${this.config.openai.model})`)
		} else {
			this.logger.warn(
				'Pipeline copilot LLM disabled — set OPENAI_API_KEY on the backend process (docker-compose backend service)',
			)
		}
	}

	async getHistory(scope: AuthContext, pipelineId: string) {
		await this.smartCity.getPipeline(scope, pipelineId)
		const session = await this.prisma.pipelineCopilotSession.findUnique({
			where: {
				organizationId_pipelineId_userId: {
					organizationId: scope.organizationId,
					pipelineId,
					userId: scope.userId,
				},
			},
			include: {
				messages: { orderBy: { createdAt: 'asc' }, take: 200 },
			},
		})
		if (!session) return []
		return session.messages.map(message => this.mapStoredMessage(message))
	}

	async resetSession(scope: AuthContext, pipelineId: string) {
		await this.smartCity.getPipeline(scope, pipelineId)
		const session = await this.prisma.pipelineCopilotSession.findUnique({
			where: {
				organizationId_pipelineId_userId: {
					organizationId: scope.organizationId,
					pipelineId,
					userId: scope.userId,
				},
			},
		})
		if (session) {
			await this.prisma.pipelineCopilotMessage.deleteMany({ where: { sessionId: session.id } })
		}
		return { reset: true }
	}

	async getContext(scope: AuthContext, pipelineId: string): Promise<PipelineCopilotContext> {
		const bundle = await this.loadBundle(scope, pipelineId)
		const chips = this.buildChips(bundle)
		const primaryVisualizations = [
			this.buildEventVolumeViz(bundle),
			this.buildExportHealthViz(bundle),
			this.buildDataFunnelViz(bundle),
		]
		return {
			generatedAt: new Date().toISOString(),
			windowHours: WINDOW_HOURS,
			pipeline: bundle.pipeline,
			runtime: bundle.runtime,
			sources: bundle.sources,
			coreUnit: bundle.coreUnit,
			exports: bundle.exports,
			dataFlow: bundle.dataFlow,
			summaryText: this.buildSummaryText(bundle),
			llmEnabled: Boolean(this.chatModel),
			chips,
			vizCommands: buildVizCommandsForChips(chips.map(chip => chip.id)),
			primaryVisualizations,
		}
	}

	async chat(
		scope: AuthContext,
		pipelineId: string,
		input: { chipId?: string; message?: string },
	): Promise<PipelineCopilotChatResponse> {
		const context = await this.getContext(scope, pipelineId)
		const bundle = await this.loadBundle(scope, pipelineId)
		const session = await this.getOrCreateSession(scope, pipelineId)
		const message = input.message?.trim() ?? ''

		const userText = message || (input.chipId
			? context.chips.find(chip => chip.id === input.chipId)?.label ?? input.chipId
			: null)

		if (userText) {
			await this.persistMessage(session.id, {
				role: 'user',
				text: userText,
				chipId: input.chipId,
				answerSource: 'system',
			})
		}

		let chipId: string | undefined
		let text = ''
		let answerSource: 'llm' | 'rules' = 'rules'
		let visualization: CopilotVizSpec | undefined

		if (input.chipId) {
			chipId = input.chipId
			visualization = this.buildVisualization(chipId, bundle)
		} else if (message) {
			const { questionText, vizChipIds } = parseComposerMessage(message)
			chipId = vizChipIds[0]
			if (!questionText && !chipId) {
				throw new BadRequestException(
					'Unknown chart command. Type / in the composer to see available charts.',
				)
			}
			if (questionText) {
				const fallback = () => this.buildOverviewNarrative(bundle)
				const answer = await this.generateChatAnswerWithLlm(context, questionText, fallback)
				text = answer.text
				answerSource = answer.answerSource
			}
			if (chipId) {
				visualization = this.buildVisualization(chipId, bundle)
			}
		}

		const saved = await this.persistMessage(session.id, {
			role: 'assistant',
			text,
			chipId,
			answerSource,
			visualization,
		})

		return {
			context,
			turn: this.mapStoredMessage(saved),
		}
	}

	async visualize(
		scope: AuthContext,
		pipelineId: string,
		chipId: string,
	): Promise<{ visualization: CopilotVizSpec; text: string }> {
		const bundle = await this.loadBundle(scope, pipelineId)
		return {
			visualization: this.buildVisualization(chipId, bundle),
			text: '',
		}
	}

	private async loadBundle(scope: AuthContext, pipelineId: string) {
		const since = new Date(Date.now() - WINDOW_HOURS * 60 * 60 * 1000)
		const pipeline = await this.smartCity.getPipeline(scope, pipelineId)
		const [sources, observability, exportTargets, exportRuns, totalEvents, previewCleaned, previewBusiness] =
			await Promise.all([
				this.smartCity.listSources(scope, pipelineId),
				this.smartCity.getObservability(scope, pipelineId),
				this.smartCity.listExportTargets(scope, pipelineId),
				this.smartCity.listExportRuns(scope, pipelineId),
				this.prisma.sensorEvent.count({
					where: { organizationId: scope.organizationId, pipelineId },
				}),
				this.smartCity.previewExportStage(scope, pipelineId, 'cleaned', 120).catch(() => null),
				this.smartCity.previewExportStage(scope, pipelineId, 'business', 120).catch(() => null),
			])

		const recentEvents = await this.prisma.sensorEvent.findMany({
			where: { organizationId: scope.organizationId, pipelineId, eventTime: { gte: since } },
			select: { eventTime: true, sensorType: true, sourceId: true },
			orderBy: { eventTime: 'asc' },
		})

		const runs24h = exportRuns.filter(run => run.createdAt >= since)
		const hourlyEvents = this.bucketCounts(
			recentEvents.map(event => event.eventTime),
			'hour',
		)
		const hourlyExportRows = this.bucketExportRows(runs24h)

		let qualitySample: PipelineCopilotContext['dataFlow']['qualitySample']
		if (previewCleaned?.rows?.length) {
			const columns = previewCleaned.columns
			const report = this.qualityService.analyzeQuality(
				previewCleaned.rows as DataRow[],
				columns,
			)
			qualitySample = {
				missingPct: Number(report.summary.missingPercentage.toFixed(1)),
				duplicatePct: Number(report.summary.duplicatePercentage.toFixed(1)),
				outlierPct: Number(report.summary.outlierPercentage.toFixed(1)),
			}
		}

		const streamConfig =
			pipeline.streamConfig && typeof pipeline.streamConfig === 'object'
				? (pipeline.streamConfig as Record<string, unknown>)
				: {}
		const runtimeConfig =
			streamConfig.runtime && typeof streamConfig.runtime === 'object'
				? (streamConfig.runtime as Record<string, unknown>)
				: {}

		const coreUnitMode = streamConfig.coreUnitMode === 'auto' ? 'auto' : 'manual'
		const autoRoutingPolicy =
			streamConfig.autoRoutingPolicy && typeof streamConfig.autoRoutingPolicy === 'object'
				? (streamConfig.autoRoutingPolicy as Record<string, unknown>)
				: null
		const lastAutoResolution =
			streamConfig.lastAutoResolution && typeof streamConfig.lastAutoResolution === 'object'
				? (streamConfig.lastAutoResolution as {
						modelId: string
						label: string
						reason: string
						sensorKind: string
						at: string
					})
				: null
		const activeModelId = pipeline.activeModelId ?? null
		const activeModelLabel =
			coreUnitMode === 'auto'
				? lastAutoResolution?.label
					? `Auto · ${lastAutoResolution.label}`
					: 'Auto routing enabled'
				: activeModelId
					? activeModelId.startsWith('hf:')
						? activeModelId.replace(/^hf:/, '')
						: activeModelId.replace(/^research:/, 'Research · ')
					: 'No model deployed'

		const exportTargetSummaries = exportTargets.map(target => ({
			id: target.id,
			name: target.name,
			stage: target.stage,
			adapterType: target.adapterType,
			status: target.status,
			isContinuous: target.isContinuous,
			cadenceSeconds: target.cadenceSeconds,
			lastRunAt: target.lastRunAt,
			lastError: target.lastError,
			recordsExportedTotal: exportRuns
				.filter(run => run.targetId === target.id && run.status === 'SUCCEEDED')
				.reduce((sum, run) => sum + (run.recordsExported ?? run.rowCount ?? 0), 0),
		}))

		const recentRunSummaries: PipelineCopilotExportRunSummary[] = exportRuns.slice(0, 8).map(run => ({
			id: run.id,
			stage: run.stage,
			adapterType: run.adapterType,
			status: run.status,
			rowCount: run.rowCount ?? 0,
			recordsExported: run.recordsExported ?? 0,
			destination: run.destination,
			createdAt: run.createdAt,
			errorMessage: run.errorMessage,
		}))

		return {
			pipeline: {
				id: pipeline.id,
				name: pipeline.name,
				status: pipeline.status,
				description: pipeline.description ?? null,
				isLive: pipeline.status === 'ACTIVE',
			},
			runtime: {
				wsConnected: pipeline.status === 'ACTIVE',
				runningSources: sources.filter(source => source.status === 'RUNNING').length,
				recentEventsBuffered: Math.min(recentEvents.length, 50),
				exportCadenceSeconds:
					typeof runtimeConfig.exportCadenceSeconds === 'number'
						? runtimeConfig.exportCadenceSeconds
						: null,
			},
			sources: {
				total: sources.length,
				running: sources.filter(source => source.status === 'RUNNING').length,
				errors: sources.filter(source => source.status === 'ERROR').length,
				items: sources.map(source => ({
					id: source.id,
					name: source.name,
					type: source.type,
					sensorKind: source.sensorKind,
					mode: source.mode,
					status: source.status,
					lastSeenAt: source.lastSeenAt,
					lastError: source.lastError,
				})),
			},
			coreUnit: {
				activeModelId,
				activeModelLabel,
				coreUnitMode,
				autoRoutingSummary:
					typeof autoRoutingPolicy?.summary === 'string' ? autoRoutingPolicy.summary : null,
				lastAutoResolution,
				autoCleaning: streamConfig.autoCleaning !== false,
				anomalyDetection: streamConfig.anomalyDetection !== false,
				schemaValidation: streamConfig.schemaValidation !== false,
			},
			exports: {
				targetCount: exportTargets.length,
				continuousCount: exportTargets.filter(target => target.isContinuous).length,
				errorTargetCount: exportTargets.filter(target => target.lastError).length,
				successRuns24h: runs24h.filter(run => run.status === 'SUCCEEDED').length,
				failedRuns24h: runs24h.filter(run => run.status === 'FAILED').length,
				totalRowsExported24h: runs24h.reduce(
					(sum, run) => sum + (run.recordsExported ?? run.rowCount ?? 0),
					0,
				),
				targets: exportTargetSummaries,
				recentRuns: recentRunSummaries,
			},
			dataFlow: {
				totalEvents,
				eventsLast24h: observability.eventsLast24h,
				lastEventAt: observability.lastEventAt,
				stageCounts: {
					raw: totalEvents,
					cleaned: previewCleaned?.rowCount ?? 0,
					business: previewBusiness?.rowCount ?? 0,
				},
				hourlyEvents,
				hourlyExportRows,
				qualitySample,
			},
			observability,
			recentEvents,
			exportRuns: runs24h,
			sourcesList: sources,
		}
	}

	private buildChips(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotChip[] {
		const chips: CopilotChip[] = [
			{ id: 'event_volume', label: 'Event volume', category: 'data', priority: 10 },
			{ id: 'export_health', label: 'Export health', category: 'exports', priority: 9 },
			{ id: 'data_funnel', label: 'Data funnel', category: 'data', priority: 8 },
			{ id: 'sources_status', label: 'Source health', category: 'sources', priority: 7 },
			{ id: 'export_adapters', label: 'By adapter', category: 'exports', priority: 6 },
			{ id: 'recent_exports', label: 'Recent exports', category: 'exports', priority: 5 },
			{ id: 'core_unit', label: 'Core unit', category: 'core', priority: 4 },
			{ id: 'data_quality', label: 'Data quality', category: 'data', priority: 3 },
		]
		if (bundle.exports.failedRuns24h > 0) {
			chips.unshift({
				id: 'export_failures',
				label: `${bundle.exports.failedRuns24h} failed exports`,
				category: 'exports',
				priority: 11,
			})
		}
		if (bundle.sources.errors > 0) {
			chips.unshift({
				id: 'source_errors',
				label: `${bundle.sources.errors} source errors`,
				category: 'sources',
				priority: 11,
			})
		}
		return chips.sort((a, b) => b.priority - a.priority).slice(0, 10)
	}

	private buildSummaryText(bundle: Awaited<ReturnType<typeof this.loadBundle>>): string {
		if (bundle.sources.total === 0 && bundle.dataFlow.eventsLast24h === 0 && bundle.exports.successRuns24h === 0) {
			return `${bundle.pipeline.name} is ready. Add ingest sources and start the pipeline — insights will populate automatically as data flows.`
		}
		const parts = [
			`${bundle.pipeline.name} is ${bundle.pipeline.isLive ? 'live' : bundle.pipeline.status.toLowerCase()}.`,
			`${bundle.sources.running}/${bundle.sources.total} sources running with ${bundle.dataFlow.eventsLast24h} events in the last 24h.`,
			`${bundle.exports.successRuns24h} export runs succeeded (${bundle.exports.totalRowsExported24h.toLocaleString()} rows) with ${bundle.exports.failedRuns24h} failures.`,
			bundle.coreUnit.activeModelId
				? `Core unit model: ${bundle.coreUnit.activeModelLabel}.`
				: 'No model is deployed on the core unit yet.',
		]
		return parts.join(' ')
	}

	private async getOrCreateSession(scope: AuthContext, pipelineId: string) {
		return this.prisma.pipelineCopilotSession.upsert({
			where: {
				organizationId_pipelineId_userId: {
					organizationId: scope.organizationId,
					pipelineId,
					userId: scope.userId,
				},
			},
			create: {
				organizationId: scope.organizationId,
				pipelineId,
				userId: scope.userId,
			},
			update: { updatedAt: new Date() },
		})
	}

	private async persistMessage(
		sessionId: string,
		input: {
			role: 'user' | 'assistant'
			text: string
			chipId?: string
			answerSource: 'llm' | 'rules' | 'system'
			visualization?: CopilotVizSpec
		},
	) {
		return this.prisma.pipelineCopilotMessage.create({
			data: {
				sessionId,
				role: input.role,
				text: input.text,
				chipId: input.chipId,
				answerSource: input.answerSource,
				visualizationJson: input.visualization ?? undefined,
			},
		})
	}

	private mapStoredMessage(message: {
		id: string
		role: string
		text: string
		chipId: string | null
		answerSource: string
		visualizationJson: unknown
	}): PipelineCopilotChatResponse['turn'] {
		const visualization =
			message.visualizationJson && typeof message.visualizationJson === 'object'
				? (message.visualizationJson as CopilotVizSpec)
				: undefined
		return {
			id: message.id,
			role: message.role === 'user' ? 'user' : 'assistant',
			text: message.text,
			chipId: message.chipId ?? undefined,
			answerSource:
				message.answerSource === 'llm' || message.answerSource === 'rules' || message.answerSource === 'system'
					? message.answerSource
					: 'rules',
			visualization,
		}
	}

	private buildRuleBasedAnswer(
		chipId: string,
		bundle: Awaited<ReturnType<typeof this.loadBundle>>,
	): string {
		return chipId === 'pipeline_overview'
			? this.buildOverviewNarrative(bundle)
			: this.buildChipNarrative(chipId, bundle)
	}

	private buildLlmContext(context: PipelineCopilotContext) {
		const { primaryVisualizations: _primaryVisualizations, chips: _chips, ...payload } = context
		return payload
	}

	private async generateChatAnswerWithLlm(
		context: PipelineCopilotContext,
		message: string,
		fallback: () => string,
	): Promise<{ text: string; answerSource: 'llm' | 'rules' }> {
		if (!this.chatModel) {
			this.logger.warn('OPENAI_API_KEY not configured, using rule-based pipeline insights answer')
			return { text: fallback(), answerSource: 'rules' }
		}

		try {
			this.logger.log('Generating pipeline insights answer with OpenAI')
			const response = await this.chatModel.invoke([
				new SystemMessage(COPILOT_SYSTEM_PROMPT),
				new HumanMessage(
					`Pipeline context (JSON):\n${JSON.stringify(this.buildLlmContext(context), null, 2)}\n\nUser question:\n${message}`,
				),
			])
			const content =
				typeof response.content === 'string'
					? response.content.trim()
					: String(response.content ?? '').trim()
			if (!content) throw new Error('Empty LLM response')
			return { text: content, answerSource: 'llm' }
		} catch (error) {
			this.logger.error('Pipeline copilot LLM answer failed, using fallback', error)
			return { text: fallback(), answerSource: 'rules' }
		}
	}

	private buildOverviewNarrative(bundle: Awaited<ReturnType<typeof this.loadBundle>>): string {
		const sourceNames = bundle.sources.items.map(source => source.name).join(', ')
		const exportSummary =
			bundle.exports.targetCount === 0
				? 'No export targets are configured yet.'
				: `${bundle.exports.targetCount} export target(s) (${bundle.exports.continuousCount} continuous) moved ${bundle.exports.totalRowsExported24h.toLocaleString()} rows in the last ${WINDOW_HOURS}h across ${bundle.exports.successRuns24h} successful runs${bundle.exports.failedRuns24h > 0 ? ` and ${bundle.exports.failedRuns24h} failures` : ''}.`

		const paragraphs = [
			`${bundle.pipeline.name} is ${bundle.pipeline.isLive ? 'live and actively streaming' : bundle.pipeline.status.toLowerCase()}. In the last ${WINDOW_HOURS}h it ingested ${bundle.dataFlow.eventsLast24h} sensor events from ${bundle.sources.total} source(s) — ${bundle.sources.running} currently running${bundle.sources.errors > 0 ? `, ${bundle.sources.errors} in error` : ''}.`,
			`These events are simulated and external sensor feeds (IoT traffic, weather, WebSocket streams) that enter the pipeline as raw records. The core unit${bundle.coreUnit.activeModelId ? ` (${bundle.coreUnit.activeModelLabel})` : ''} validates, cleans, and scores them before they land in the medallion lake: ${bundle.dataFlow.stageCounts.raw.toLocaleString()} raw stored, ${bundle.dataFlow.stageCounts.cleaned.toLocaleString()} cleaned preview rows, ${bundle.dataFlow.stageCounts.business.toLocaleString()} business preview rows.`,
			exportSummary,
		]

		if (sourceNames) {
			paragraphs.push(`Configured sources: ${sourceNames}.`)
		}

		if (!bundle.pipeline.isLive && bundle.sources.running === 0) {
			paragraphs.push(
				'The pipeline is paused right now — start it from the workbench to resume live ingestion and continuous exports.',
			)
		}

		return paragraphs.join('\n\n')
	}

	private buildChipNarrative(chipId: string, bundle: Awaited<ReturnType<typeof this.loadBundle>>): string {
		switch (chipId) {
			case 'pipeline_overview':
				return this.buildOverviewNarrative(bundle)
			case 'event_volume':
				return `In the last ${WINDOW_HOURS}h this pipeline captured ${bundle.dataFlow.eventsLast24h} sensor events${bundle.dataFlow.lastEventAt ? ` (last at ${new Date(bundle.dataFlow.lastEventAt).toLocaleString()}).` : '.'}`
			case 'export_health':
				return `${bundle.exports.successRuns24h} export runs succeeded and ${bundle.exports.failedRuns24h} failed in the last ${WINDOW_HOURS}h, moving ${bundle.exports.totalRowsExported24h.toLocaleString()} rows downstream.`
			case 'export_failures':
				return `${bundle.exports.failedRuns24h} export runs failed recently. Check targets with errors and inspect the latest run messages below.`
			case 'data_funnel':
				return `Medallion flow snapshot: ${bundle.dataFlow.stageCounts.raw.toLocaleString()} raw events stored, ${bundle.dataFlow.stageCounts.cleaned.toLocaleString()} cleaned preview rows, ${bundle.dataFlow.stageCounts.business.toLocaleString()} business preview rows.`
			case 'sources_status':
				return `${bundle.sources.running} sources are running, ${bundle.sources.errors} in error, out of ${bundle.sources.total} configured ingest feeds.`
			case 'source_errors':
				return `${bundle.sources.errors} source(s) report errors. Open the Sources stage to inspect endpoints and last error messages.`
			case 'export_adapters':
				return `Export activity in the last ${WINDOW_HOURS}h broken down by destination adapter.`
			case 'recent_exports':
				return `Latest export runs with row counts, adapters, and status.`
			case 'core_unit':
				return bundle.coreUnit.coreUnitMode === 'auto'
					? `Core unit is in auto mode${bundle.coreUnit.autoRoutingSummary ? `: ${bundle.coreUnit.autoRoutingSummary}` : ''}. Last routed model: ${bundle.coreUnit.activeModelLabel}.`
					: bundle.coreUnit.activeModelId
						? `Core unit is configured with ${bundle.coreUnit.activeModelLabel}. Guards: auto-clean ${bundle.coreUnit.autoCleaning ? 'on' : 'off'}, schema ${bundle.coreUnit.schemaValidation ? 'on' : 'off'}.`
						: 'No inference model is selected. Deploy a Hugging Face or research checkpoint to produce business-stage rows.'
			case 'data_quality':
				return bundle.dataFlow.qualitySample
					? `Cleaned-stage sample quality: ${bundle.dataFlow.qualitySample.missingPct}% missing, ${bundle.dataFlow.qualitySample.duplicatePct}% duplicates, ${bundle.dataFlow.qualitySample.outlierPct}% outliers.`
					: 'Not enough cleaned preview rows yet to compute a quality sample. Start the pipeline and wait for events.'
			default:
				return this.buildSummaryText(bundle)
		}
	}

	private buildVisualization(chipId: string, bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		switch (chipId) {
			case 'pipeline_overview':
			case 'data_funnel':
				return this.buildDataFunnelViz(bundle)
			case 'export_health':
			case 'export_failures':
				return this.buildExportHealthViz(bundle)
			case 'sources_status':
			case 'source_errors':
				return this.buildSourcesViz(bundle)
			case 'export_adapters':
				return this.buildExportAdapterViz(bundle)
			case 'recent_exports':
				return this.buildRecentExportsViz(bundle)
			case 'core_unit':
				return this.buildCoreUnitViz(bundle)
			case 'data_quality':
				return this.buildQualityViz(bundle)
			case 'event_volume':
			default:
				return this.buildEventVolumeViz(bundle)
		}
	}

	private buildEventVolumeViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		const labels = bundle.dataFlow.hourlyEvents.map(bucket => bucket.hour)
		return {
			id: 'event_volume',
			title: 'Event volume (24h)',
			subtitle: 'Sensor events per UTC hour',
			type: 'area',
			labels,
			series: [{ name: 'Events', values: bundle.dataFlow.hourlyEvents.map(bucket => bucket.count), color: '#34d0c3' }],
		}
	}

	private buildExportHealthViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		return {
			id: 'export_health',
			title: 'Export run outcomes (24h)',
			type: 'donut',
			labels: ['Succeeded', 'Failed', 'Other'],
			series: [
				{
					name: 'Runs',
					values: [
						bundle.exports.successRuns24h,
						bundle.exports.failedRuns24h,
						Math.max(
							0,
							bundle.exportRuns.length -
								bundle.exports.successRuns24h -
								bundle.exports.failedRuns24h,
						),
					],
					color: '#34d0c3',
				},
			],
			kpis: [
				{ label: 'Rows exported', value: bundle.exports.totalRowsExported24h.toLocaleString() },
				{ label: 'Continuous targets', value: bundle.exports.continuousCount },
			],
		}
	}

	private buildDataFunnelViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		const stages = ['Raw events', 'Cleaned preview', 'Business preview']
		const values = [
			bundle.dataFlow.stageCounts.raw,
			bundle.dataFlow.stageCounts.cleaned,
			bundle.dataFlow.stageCounts.business,
		]
		return {
			id: 'data_funnel',
			title: 'Medallion data funnel',
			subtitle: 'Volume at each pipeline stage',
			type: 'funnel',
			labels: stages,
			series: [{ name: 'Rows', values, color: '#7c8cff' }],
		}
	}

	private buildSourcesViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		const statusCounts = {
			RUNNING: bundle.sources.running,
			ERROR: bundle.sources.errors,
			OTHER: Math.max(0, bundle.sources.total - bundle.sources.running - bundle.sources.errors),
		}
		return {
			id: 'sources_status',
			title: 'Ingest source health',
			type: 'bar',
			labels: ['Running', 'Error', 'Idle/Other'],
			series: [{ name: 'Sources', values: [statusCounts.RUNNING, statusCounts.ERROR, statusCounts.OTHER], color: '#34d0c3' }],
			rows: bundle.sources.items.slice(0, 6).map(source => ({
				name: source.name,
				type: source.type,
				status: source.status,
				sensor: source.sensorKind,
			})),
		}
	}

	private buildExportAdapterViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		const totals = new Map<string, number>()
		for (const run of bundle.exportRuns) {
			if (run.status !== 'SUCCEEDED') continue
			totals.set(run.adapterType, (totals.get(run.adapterType) ?? 0) + (run.recordsExported ?? run.rowCount ?? 0))
		}
		const labels = [...totals.keys()]
		return {
			id: 'export_adapters',
			title: 'Rows exported by adapter (24h)',
			type: 'bar',
			labels: labels.length ? labels : ['No successful exports'],
			series: [
				{
					name: 'Rows',
					values: labels.length ? labels.map(label => totals.get(label) ?? 0) : [0],
					color: '#7c8cff',
				},
			],
		}
	}

	private buildRecentExportsViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		return {
			id: 'recent_exports',
			title: 'Recent export runs',
			type: 'table',
			labels: [],
			series: [],
			rows: bundle.exports.recentRuns.map(run => ({
				stage: run.stage,
				adapter: run.adapterType,
				status: run.status,
				rows: run.recordsExported || run.rowCount,
				at: new Date(run.createdAt).toLocaleString(),
			})),
		}
	}

	private buildCoreUnitViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		return {
			id: 'core_unit',
			title: 'Core unit status',
			type: 'kpi',
			labels: [],
			series: [],
			kpis: [
				{ label: 'Active model', value: bundle.coreUnit.activeModelLabel },
				{ label: 'Auto cleaning', value: bundle.coreUnit.autoCleaning ? 'Enabled' : 'Disabled' },
				{ label: 'Schema validation', value: bundle.coreUnit.schemaValidation ? 'Enabled' : 'Disabled' },
				{ label: 'Anomaly detection', value: bundle.coreUnit.anomalyDetection ? 'Enabled' : 'Disabled' },
			],
		}
	}

	private buildQualityViz(bundle: Awaited<ReturnType<typeof this.loadBundle>>): CopilotVizSpec {
		const sample = bundle.dataFlow.qualitySample
		return {
			id: 'data_quality',
			title: 'Cleaned-stage quality sample',
			subtitle: sample ? 'Based on latest cleaned preview rows' : 'Waiting for cleaned data',
			type: 'bar',
			labels: sample ? ['Missing %', 'Duplicates %', 'Outliers %'] : ['No sample yet'],
			series: [
				{
					name: 'Rate',
					values: sample ? [sample.missingPct, sample.duplicatePct, sample.outlierPct] : [0],
					color: '#f2c14f',
				},
			],
		}
	}

	private bucketCounts(dates: Date[], granularity: 'hour'): Array<{ hour: string; count: number }> {
		const counts = new Map<string, number>()
		for (const date of dates) {
			const key = `${String(date.getUTCHours()).padStart(2, '0')}:00`
			counts.set(key, (counts.get(key) ?? 0) + 1)
		}
		return this.fillRecentHourBuckets(counts, 12)
	}

	private bucketExportRows(
		runs: Array<{ createdAt: Date | string; recordsExported?: number | null; rowCount?: number | null }>,
	) {
		const counts = new Map<string, number>()
		for (const run of runs) {
			const hour = `${String(new Date(run.createdAt).getUTCHours()).padStart(2, '0')}:00`
			counts.set(hour, (counts.get(hour) ?? 0) + (run.recordsExported ?? run.rowCount ?? 0))
		}
		return this.fillRecentHourBuckets(counts, 12).map(({ hour, count }) => ({ hour, rows: count }))
	}

	private fillRecentHourBuckets(counts: Map<string, number>, slots: number) {
		const result: Array<{ hour: string; count: number }> = []
		const now = new Date()
		for (let offset = slots - 1; offset >= 0; offset -= 1) {
			const slot = new Date(now.getTime() - offset * 60 * 60 * 1000)
			const hour = `${String(slot.getUTCHours()).padStart(2, '0')}:00`
			result.push({ hour, count: counts.get(hour) ?? 0 })
		}
		return result
	}

	async getAnalysisBundle(scope: AuthContext, pipelineId: string) {
		await this.smartCity.getPipeline(scope, pipelineId)
		return this.loadBundle(scope, pipelineId)
	}

	buildStandardVisualization(
		chipId: string,
		bundle: Awaited<ReturnType<typeof this.loadBundle>>,
	): CopilotVizSpec {
		return this.buildVisualization(chipId, bundle)
	}
}
