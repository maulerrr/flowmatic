import { Injectable, Logger } from '@nestjs/common'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { ChatOpenAI } from '@langchain/openai'
import { AppConfigService } from 'src/common/config/config.service'
import { PipelineModelRegistryService } from './pipeline-model-registry.service'
import { PipelineModelRouterService } from './pipeline-model-router.service'
import { AutoRoutingPolicy } from './pipeline-model-router.types'

const AUTO_ROUTING_PROMPT = `You are Flowmatic Core Unit Auto-Routing Planner.

Given:
- sensor kinds configured on a smart-city pipeline
- a deterministic list of allowed trained model bindings (modelId must be copied exactly)
- registry metadata (modality, dataset, task)

Return ONLY JSON:
{
  "summary": "1-2 sentences for operators",
  "bindings": [
    { "sensorKind": "traffic", "modelId": "research:...", "label": "...", "reason": "..." }
  ]
}

Rules:
- NEVER route weather models to traffic sensors or geospatial traffic feeds.
- NEVER route traffic forecasters to weather-only sensors.
- Prefer anomaly models when sensorKind is traffic and quality guards are enabled.
- Keep bindings subset of allowed bindings; you may drop incompatible ones but must not invent model IDs.`

@Injectable()
export class PipelineAutoRoutingService {
	private readonly logger = new Logger(PipelineAutoRoutingService.name)
	private readonly chatModel?: ChatOpenAI

	constructor(
		private readonly config: AppConfigService,
		private readonly registry: PipelineModelRegistryService,
		private readonly router: PipelineModelRouterService,
	) {
		if (this.config.openai.apiKey) {
			this.chatModel = new ChatOpenAI({
				modelName: this.config.openai.model,
				temperature: 0.1,
				openAIApiKey: this.config.openai.apiKey,
			})
		}
	}

	async buildPolicy(sensorKinds: string[]): Promise<AutoRoutingPolicy> {
		const rulePolicy = this.router.buildDefaultPolicy(sensorKinds)
		if (!this.chatModel || rulePolicy.bindings.length === 0) return rulePolicy

		try {
			const allowed = rulePolicy.bindings
			const registry = this.registry.listModels().slice(0, 40)
			const response = await this.chatModel.invoke([
				new SystemMessage(AUTO_ROUTING_PROMPT),
				new HumanMessage(
					JSON.stringify(
						{
							sensorKinds,
							allowedBindings: allowed,
							registry: registry.map(entry => ({
								id: entry.id,
								kind: entry.kind,
								dataset: entry.dataset,
								modality: entry.modality,
								tasks: entry.tasks,
								sensorKinds: entry.sensorKinds,
							})),
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
				summary?: string
				bindings?: AutoRoutingPolicy['bindings']
			}
			const allowedIds = new Set(allowed.map(binding => binding.modelId))
			const bindings = (parsed.bindings ?? [])
				.filter(binding => allowedIds.has(binding.modelId))
				.map(binding => {
					const fallback = allowed.find(item => item.modelId === binding.modelId)
					return {
						sensorKind: binding.sensorKind,
						modelId: binding.modelId,
						label: binding.label || fallback?.label || binding.modelId,
						reason: binding.reason || fallback?.reason || 'LLM-selected binding',
					}
				})

			if (!bindings.length) return rulePolicy

			return {
				generatedAt: new Date().toISOString(),
				plannerSource: 'llm',
				summary: parsed.summary?.trim() || rulePolicy.summary,
				bindings,
			}
		} catch (error) {
			this.logger.warn('Auto routing LLM planner failed, using rules', error)
			return rulePolicy
		}
	}
}
