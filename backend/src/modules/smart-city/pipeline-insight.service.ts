import { BadRequestException, Injectable, Logger, NotFoundException } from '@nestjs/common'
import { DataRow } from 'src/common/types/data.types'
import { AuthContext } from 'src/modules/auth/auth-context.service'
import { PrismaService } from 'src/prisma/prisma.service'
import { PipelineAnalysisEngineService } from './pipeline-analysis-engine.service'
import { PipelineCopilotService } from './pipeline-copilot.service'
import { PipelineDataProfilerService } from './pipeline-data-profiler.service'
import { PipelineInsightPlannerService } from './pipeline-insight-planner.service'
import { PipelineInsightVizBuilderService } from './pipeline-insight-viz-builder.service'
import {
	InsightDepth,
	InsightFocus,
	InsightTrigger,
	PipelineInsightConfig,
	PipelineInsightRunSummary,
} from './pipeline-insight.types'
import { SmartCityService } from './smart-city.service'

function flattenRecord(input: Record<string, unknown>, prefix = ''): Record<string, unknown> {
	const flat: Record<string, unknown> = {}
	for (const [key, value] of Object.entries(input)) {
		const path = prefix ? `${prefix}.${key}` : key
		if (value && typeof value === 'object' && !Array.isArray(value)) {
			Object.assign(flat, flattenRecord(value as Record<string, unknown>, path))
		} else {
			flat[path] = value
		}
	}
	return flat
}

@Injectable()
export class PipelineInsightService {
	private readonly logger = new Logger(PipelineInsightService.name)

	constructor(
		private readonly prisma: PrismaService,
		private readonly smartCity: SmartCityService,
		private readonly copilot: PipelineCopilotService,
		private readonly profiler: PipelineDataProfilerService,
		private readonly analysisEngine: PipelineAnalysisEngineService,
		private readonly planner: PipelineInsightPlannerService,
		private readonly vizBuilder: PipelineInsightVizBuilderService,
	) {}

	async getConfig(scope: AuthContext, pipelineId: string): Promise<PipelineInsightConfig> {
		const pipeline = await this.smartCity.getPipeline(scope, pipelineId)
		const intervalMinutes = pipeline.insightIntervalMinutes ?? 0
		const lastRunAt = pipeline.lastInsightRunAt?.toISOString() ?? null
		return {
			intervalMinutes,
			depth: (pipeline.insightDepth as InsightDepth) ?? 'standard',
			focus: (pipeline.insightFocus as InsightFocus) ?? 'all',
			enabled: intervalMinutes > 0,
			lastRunAt,
			nextRunAt:
				intervalMinutes > 0 && lastRunAt
					? new Date(new Date(lastRunAt).getTime() + intervalMinutes * 60_000).toISOString()
					: null,
		}
	}

	async updateConfig(
		scope: AuthContext,
		pipelineId: string,
		input: { intervalMinutes?: number; depth?: InsightDepth; focus?: InsightFocus },
	): Promise<PipelineInsightConfig> {
		if (input.intervalMinutes !== undefined && ![0, 15, 30, 60, 360, 1440].includes(input.intervalMinutes)) {
			throw new BadRequestException('intervalMinutes must be one of 0, 15, 30, 60, 360, 1440')
		}
		await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId, organizationId: scope.organizationId },
			data: {
				...(input.intervalMinutes !== undefined
					? { insightIntervalMinutes: input.intervalMinutes }
					: {}),
				...(input.depth ? { insightDepth: input.depth } : {}),
				...(input.focus ? { insightFocus: input.focus } : {}),
			},
		})
		return this.getConfig(scope, pipelineId)
	}

	async listRuns(scope: AuthContext, pipelineId: string, limit = 20): Promise<PipelineInsightRunSummary[]> {
		await this.smartCity.getPipeline(scope, pipelineId)
		const runs = await this.prisma.pipelineInsightRun.findMany({
			where: { organizationId: scope.organizationId, pipelineId },
			orderBy: { createdAt: 'desc' },
			take: limit,
		})
		return runs.map(run => this.mapRun(run))
	}

	async getRun(scope: AuthContext, pipelineId: string, runId: string): Promise<PipelineInsightRunSummary> {
		await this.smartCity.getPipeline(scope, pipelineId)
		const run = await this.prisma.pipelineInsightRun.findFirst({
			where: { id: runId, pipelineId, organizationId: scope.organizationId },
		})
		if (!run) throw new NotFoundException('Insight run not found')
		return this.mapRun(run)
	}

	async triggerRun(scope: AuthContext, pipelineId: string): Promise<PipelineInsightRunSummary> {
		return this.executeRun(scope, pipelineId, 'manual')
	}

	async executeRun(
		scope: AuthContext,
		pipelineId: string,
		trigger: InsightTrigger,
	): Promise<PipelineInsightRunSummary> {
		const pipeline = await this.smartCity.getPipeline(scope, pipelineId)
		const depth = (pipeline.insightDepth as InsightDepth) ?? 'standard'
		const focus = (pipeline.insightFocus as InsightFocus) ?? 'all'

		const run = await this.prisma.pipelineInsightRun.create({
			data: {
				organizationId: scope.organizationId,
				pipelineId,
				trigger,
				status: 'RUNNING',
				intervalMinutes: pipeline.insightIntervalMinutes ?? 0,
				depth,
				focus,
			},
		})

		try {
			const [bundle, context, eventSample, previewCleaned] = await Promise.all([
				this.copilot.getAnalysisBundle(scope, pipelineId),
				this.copilot.getContext(scope, pipelineId),
				this.prisma.sensorEvent.findMany({
					where: { organizationId: scope.organizationId, pipelineId },
					orderBy: { eventTime: 'desc' },
					take: depth === 'deep' ? 500 : depth === 'standard' ? 250 : 120,
					select: { payloadJson: true, sensorType: true, location: true },
				}),
				this.smartCity.previewExportStage(scope, pipelineId, 'cleaned', 200).catch(() => null),
			])

			const eventPayloads = eventSample.map(event => {
				const payload =
					event.payloadJson && typeof event.payloadJson === 'object'
						? (event.payloadJson as Record<string, unknown>)
						: {}
				if (event.location) payload.location = event.location
				return payload
			})

			const profile = this.profiler.profileMixedSamples({
				eventPayloads,
				cleanedRows: (previewCleaned?.rows ?? []) as DataRow[],
				sensorTypes: [...new Set(eventSample.map(event => event.sensorType))],
				sourceCount: bundle.sources.total,
			})

			const { geoPoints, scatterPair, numericPairs } = this.extractAnalysisPairs(
				eventPayloads,
				(previewCleaned?.rows ?? []) as DataRow[],
				profile,
			)

			const findings = this.analysisEngine.analyze({
				profile,
				eventsLast24h: bundle.dataFlow.eventsLast24h,
				exportSuccess24h: bundle.exports.successRuns24h,
				exportFailed24h: bundle.exports.failedRuns24h,
				sourceErrors: bundle.sources.errors,
				runningSources: bundle.sources.running,
				totalSources: bundle.sources.total,
				hourlyEvents: bundle.dataFlow.hourlyEvents,
				qualitySample: bundle.dataFlow.qualitySample,
				geoPoints,
				numericPairs,
			})

			const plan = await this.planner.plan({ profile, findings, context, depth, focus })
			const visualizations = this.vizBuilder.buildAll(
				plan.visualizationIds,
				{
					dataFlow: bundle.dataFlow,
					exports: bundle.exports,
					sources: bundle.sources,
					coreUnit: bundle.coreUnit,
					buildStandardViz: chipId => this.copilot.buildStandardVisualization(chipId, bundle),
				},
				profile,
				findings,
				geoPoints,
				scatterPair,
			)

			const saved = await this.prisma.pipelineInsightRun.update({
				where: { id: run.id },
				data: {
					status: 'SUCCEEDED',
					profileJson: profile,
					findingsJson: findings,
					planJson: plan,
					visualizationsJson: visualizations,
					narrative: plan.narrative,
					finishedAt: new Date(),
				},
			})

			await this.prisma.smartCityPipeline.update({
				where: { id: pipelineId },
				data: { lastInsightRunAt: new Date() },
			})

			return this.mapRun(saved)
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Insight run failed'
			this.logger.error(`Insight run failed for pipeline ${pipelineId}`, error)
			const saved = await this.prisma.pipelineInsightRun.update({
				where: { id: run.id },
				data: {
					status: 'FAILED',
					errorMessage: message,
					finishedAt: new Date(),
				},
			})
			return this.mapRun(saved)
		}
	}

	async hasRunningInsightRun(pipelineId: string) {
		const running = await this.prisma.pipelineInsightRun.findFirst({
			where: { pipelineId, status: 'RUNNING' },
			select: { id: true },
		})
		return Boolean(running)
	}

	async listDuePipelines() {
		const pipelines = await this.prisma.smartCityPipeline.findMany({
			where: {
				insightIntervalMinutes: { gt: 0 },
				status: 'ACTIVE',
			},
			select: {
				id: true,
				organizationId: true,
				createdByUserId: true,
				insightIntervalMinutes: true,
				lastInsightRunAt: true,
			},
		})
		const now = Date.now()
		return pipelines.filter(pipeline => {
			if (!pipeline.lastInsightRunAt) return true
			const elapsed = now - pipeline.lastInsightRunAt.getTime()
			return elapsed >= (pipeline.insightIntervalMinutes ?? 0) * 60_000
		})
	}

	private extractAnalysisPairs(
		eventPayloads: Array<Record<string, unknown>>,
		cleanedRows: DataRow[],
		profile: import('./pipeline-insight.types').PipelineDataProfile,
	) {
		const flattenedEvents = eventPayloads.map(row => flattenRecord(row))
		const flattenedCleaned = cleanedRows.map(row => flattenRecord(row as Record<string, unknown>))
		const merged = [...flattenedEvents, ...flattenedCleaned]

		const geoPoints: Array<{ lat: number; lng: number; weight?: number }> = []
		if (profile.latField && profile.lngField) {
			for (const row of merged) {
				const lat = Number(row[profile.latField])
				const lng = Number(row[profile.lngField])
				if (Number.isFinite(lat) && Number.isFinite(lng)) geoPoints.push({ lat, lng, weight: 1 })
			}
		}

		const numericFields = profile.fields
			.filter(field => field.kind === 'numeric')
			.map(field => field.name)
			.filter(
				(name, index, names) =>
					names.findIndex(
						candidate => candidate.toLowerCase().replace(/[^a-z0-9]/g, '') === name.toLowerCase().replace(/[^a-z0-9]/g, ''),
					) === index,
			)
			.slice(0, 6)
		const numericPairs: Array<{
			xField: string
			yField: string
			points: Array<{ x: number; y: number }>
		}> = []
		let scatterPair: { xField: string; yField: string; points: Array<{ x: number; y: number }> } | undefined

		for (let i = 0; i < numericFields.length; i += 1) {
			for (let j = i + 1; j < numericFields.length; j += 1) {
				const xField = numericFields[i]
				const yField = numericFields[j]
				const xKey = xField.toLowerCase().replace(/[^a-z0-9]/g, '')
				const yKey = yField.toLowerCase().replace(/[^a-z0-9]/g, '')
				if (xKey === yKey) continue
				const points = merged
					.map(row => ({ x: Number(row[xField]), y: Number(row[yField]) }))
					.filter(point => Number.isFinite(point.x) && Number.isFinite(point.y))
				if (points.length >= 8) {
					numericPairs.push({ xField, yField, points })
					if (!scatterPair) scatterPair = { xField, yField, points }
				}
			}
		}

		return { geoPoints, scatterPair, numericPairs }
	}

	private mapRun(run: {
		id: string
		pipelineId: string
		trigger: string
		status: string
		depth: string
		focus: string
		profileJson: unknown
		findingsJson: unknown
		planJson: unknown
		visualizationsJson: unknown
		narrative: string | null
		errorMessage: string | null
		startedAt: Date
		finishedAt: Date | null
		createdAt: Date
	}): PipelineInsightRunSummary {
		const plan = (run.planJson ?? {}) as {
			headline?: string
			findings?: PipelineInsightRunSummary['findings']
			actions?: PipelineInsightRunSummary['actions']
			plannerSource?: 'llm' | 'rules'
		}
		const findingsPayload = (run.findingsJson ?? {}) as { findings?: PipelineInsightRunSummary['findings'] }
		return {
			id: run.id,
			pipelineId: run.pipelineId,
			trigger: run.trigger as InsightTrigger,
			status: run.status,
			depth: run.depth as InsightDepth,
			focus: run.focus as InsightFocus,
			headline: plan.headline ?? 'Pipeline insight run',
			narrative: run.narrative ?? '',
			profile: (run.profileJson ?? {}) as PipelineInsightRunSummary['profile'],
			findings: plan.findings ?? findingsPayload.findings ?? [],
			actions: plan.actions ?? [],
			visualizations: Array.isArray(run.visualizationsJson)
				? (run.visualizationsJson as PipelineInsightRunSummary['visualizations'])
				: [],
			plannerSource: plan.plannerSource ?? 'rules',
			startedAt: run.startedAt.toISOString(),
			finishedAt: run.finishedAt?.toISOString() ?? null,
			createdAt: run.createdAt.toISOString(),
			errorMessage: run.errorMessage,
		}
	}
}
