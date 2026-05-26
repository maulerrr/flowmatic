import { Injectable, Logger } from '@nestjs/common'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { AppConfigService } from 'src/common/config/config.service'
import {
	INSIGHT_VIZ_CATALOG,
	InsightAction,
	InsightDepth,
	InsightFinding,
	InsightFocus,
	InsightVizId,
	PipelineAnalysisFindings,
	PipelineDataProfile,
	PipelineInsightPlan,
} from './pipeline-insight.types'
import { PipelineCopilotContext } from './pipeline-copilot.types'

const PLANNER_SYSTEM_PROMPT = `You are Flowmatic Pipeline Insight Planner for smart-city data pipelines.

You receive:
1) A data profile (field types, domains like geospatial/timeseries)
2) Deterministic analysis findings (patterns, correlations, ops alerts)
3) Pipeline operational context

Return ONLY valid JSON matching this schema:
{
  "headline": "short title",
  "narrative": "2-4 paragraphs plain language, grounded in evidence only",
  "visualizationIds": ["event_volume", "geo_map", ...],
  "highlightFindingIds": ["finding-id", ...],
  "actions": [{"id":"action-1","label":"...","description":"...","priority":"high|medium|low","kind":"ops|quality|export|source|model"}]
}

Rules:
- Pick visualizationIds ONLY from the allowed catalog ids provided.
- If geospatial domain is present, prefer geo_map and/or geo_heatmap.
- If correlations exist, include metric_scatter when appropriate.
- Never invent metrics not present in findings/context.
- Max 6 visualizations.`

@Injectable()
export class PipelineInsightPlannerService {
	private readonly logger = new Logger(PipelineInsightPlannerService.name)
	private readonly chatModel?: ChatOpenAI

	constructor(private readonly config: AppConfigService) {
		if (this.config.openai.apiKey) {
			this.chatModel = new ChatOpenAI({
				modelName: this.config.openai.model,
				temperature: 0.2,
				openAIApiKey: this.config.openai.apiKey,
			})
		}
	}

	async plan(input: {
		profile: PipelineDataProfile
		findings: PipelineAnalysisFindings
		context: PipelineCopilotContext
		depth: InsightDepth
		focus: InsightFocus
	}): Promise<PipelineInsightPlan> {
		const allowedIds = INSIGHT_VIZ_CATALOG.map(item => item.id)
		if (this.chatModel) {
			try {
				const response = await this.chatModel.invoke([
					new SystemMessage(PLANNER_SYSTEM_PROMPT),
					new HumanMessage(
						JSON.stringify(
							{
								allowedVisualizationIds: allowedIds,
								depth: input.depth,
								focus: input.focus,
								profile: input.profile,
								findings: input.findings,
								pipelineContext: {
									pipeline: input.context.pipeline,
									sources: input.context.sources,
									exports: input.context.exports,
									dataFlow: input.context.dataFlow,
									coreUnit: input.context.coreUnit,
								},
							},
							null,
							2,
						),
					),
				])
				const raw =
					typeof response.content === 'string'
						? response.content.trim()
						: String(response.content ?? '').trim()
				const parsed = JSON.parse(raw.replace(/^```json\s*|```$/g, '').trim()) as {
					headline?: string
					narrative?: string
					visualizationIds?: string[]
					highlightFindingIds?: string[]
					actions?: InsightAction[]
				}
				const visualizationIds = (parsed.visualizationIds ?? [])
					.filter((id): id is InsightVizId => allowedIds.includes(id as InsightVizId))
					.slice(0, 6)
				const highlightIds = new Set(parsed.highlightFindingIds ?? [])
				const findings = input.findings.findings.filter(
					finding => highlightIds.size === 0 || highlightIds.has(finding.id),
				)
				return {
					headline: parsed.headline?.trim() || this.buildHeadline(input.findings.findings),
					narrative: parsed.narrative?.trim() || this.buildRuleNarrative(input),
					findings: findings.length ? findings : input.findings.findings.slice(0, 5),
					visualizationIds: visualizationIds.length
						? visualizationIds
						: this.buildRuleVisualizationIds(input.profile, input.findings),
					actions: (parsed.actions ?? []).slice(0, 5),
					plannerSource: 'llm',
				}
			} catch (error) {
				this.logger.warn('Insight planner LLM failed, using rules', error)
			}
		}
		return this.buildRulePlan(input)
	}

	private buildRulePlan(input: {
		profile: PipelineDataProfile
		findings: PipelineAnalysisFindings
		context: PipelineCopilotContext
	}): PipelineInsightPlan {
		const findings = input.findings.findings.slice(0, 6)
		const actions = this.buildRuleActions(input.findings.findings, input.context)
		return {
			headline: this.buildHeadline(findings),
			narrative: this.buildRuleNarrative(input),
			findings,
			visualizationIds: this.buildRuleVisualizationIds(input.profile, input.findings),
			actions,
			plannerSource: 'rules',
		}
	}

	private buildHeadline(findings: InsightFinding[]) {
		const critical = findings.find(finding => finding.severity === 'critical')
		if (critical) return critical.title
		const warning = findings.find(finding => finding.severity === 'warning')
		if (warning) return warning.title
		return findings[0]?.title ?? 'Pipeline insight snapshot'
	}

	private buildRuleNarrative(input: {
		profile: PipelineDataProfile
		findings: PipelineAnalysisFindings
		context: PipelineCopilotContext
	}) {
		const paragraphs = [
			`${input.context.pipeline.name} analysis sampled ${input.profile.sampleSize} rows across domains: ${input.profile.domains.join(', ') || 'general telemetry'}.`,
			...input.findings.findings.slice(0, 4).map(finding => finding.summary),
		]
		if (input.findings.featureEngineering.length) {
			paragraphs.push(
				`Suggested engineered features: ${input.findings.featureEngineering.map(item => item.name).join(', ')}.`,
			)
		}
		return paragraphs.join('\n\n')
	}

	private buildRuleVisualizationIds(
		profile: PipelineDataProfile,
		findings: PipelineAnalysisFindings,
	): InsightVizId[] {
		const ids = new Set<InsightVizId>(['event_volume', 'data_funnel', 'export_health'])
		if (profile.hasGeospatial) {
			ids.add('geo_map')
			if (findings.geoHotspots.length >= 2) ids.add('geo_heatmap')
		}
		if (findings.correlations.length) ids.add('metric_scatter')
		if (findings.findings.some(finding => finding.category === 'quality')) ids.add('quality_kpis')
		if (findings.findings.some(finding => finding.category === 'ops')) ids.add('source_health')
		return [...ids].slice(0, 6)
	}

	private buildRuleActions(findings: InsightFinding[], context: PipelineCopilotContext): InsightAction[] {
		const actions: InsightAction[] = []
		if (findings.some(finding => finding.id === 'export-failures')) {
			actions.push({
				id: 'review-exports',
				label: 'Review failing export targets',
				description: 'Open export settings and inspect the latest failed run messages.',
				priority: 'high',
				kind: 'export',
			})
		}
		if (findings.some(finding => finding.id === 'source-errors')) {
			actions.push({
				id: 'fix-sources',
				label: 'Inspect erroring ingest sources',
				description: 'Verify endpoints and restart sources reporting errors.',
				priority: 'high',
				kind: 'source',
			})
		}
		if (findings.some(finding => finding.category === 'quality')) {
			actions.push({
				id: 'tune-cleaning',
				label: 'Tune cleaning guards',
				description: 'Review missing/outlier rates and adjust auto-cleaning thresholds.',
				priority: 'medium',
				kind: 'quality',
			})
		}
		if (!context.coreUnit.activeModelId) {
			actions.push({
				id: 'deploy-model',
				label: 'Deploy a core unit model',
				description: 'Select a Hugging Face or research checkpoint to enrich business-stage rows.',
				priority: 'medium',
				kind: 'model',
			})
		}
		return actions.slice(0, 5)
	}
}
