import { Injectable } from '@nestjs/common'
import { PipelineModelRegistryService } from './pipeline-model-registry.service'
import {
	AutoRoutingBinding,
	AutoRoutingPolicy,
	CoreUnitMode,
	EventRoutingProfile,
	ModelModality,
	ModelRegistryEntry,
	ModelRoutingDecision,
	ModelTask,
} from './pipeline-model-router.types'

const TRAFFIC_FIELD_HINTS = [
	'speed',
	'traffic',
	'vehicle',
	'density',
	'congestion',
	'flow',
	'occupancy',
]
const WEATHER_FIELD_HINTS = ['temperature', 'humidity', 'pressure', 'wind', 'precipitation', 'weather']

@Injectable()
export class PipelineModelRouterService {
	constructor(private readonly registry: PipelineModelRegistryService) {}

	classifyEvent(sensorKind: string, payload: Record<string, unknown>): EventRoutingProfile {
		const fieldNames = Object.keys(payload)
		const normalized = fieldNames.map(name => name.toLowerCase())
		const hasGeo = normalized.some(
			name =>
				name.includes('lat') ||
				name.includes('lng') ||
				name.includes('longitude') ||
				name.includes('latitude'),
		)
		const hasWeatherFields = normalized.some(name =>
			WEATHER_FIELD_HINTS.some(hint => name.includes(hint)),
		)
		const hasTrafficFields = normalized.some(name =>
			TRAFFIC_FIELD_HINTS.some(hint => name.includes(hint)),
		)

		let modality: ModelModality = this.inferModality(sensorKind, hasWeatherFields, hasTrafficFields)
		if (sensorKind === 'weather' || (hasWeatherFields && !hasTrafficFields)) modality = 'weather'
		if (sensorKind === 'traffic' || hasTrafficFields) modality = 'traffic'
		if (sensorKind === 'air') modality = hasWeatherFields ? 'weather' : 'generic'

		const preferredTask = this.inferPreferredTask(payload, hasTrafficFields, hasWeatherFields)

		return {
			sensorKind: sensorKind || 'generic',
			modality,
			hasGeo,
			hasWeatherFields,
			hasTrafficFields,
			preferredTask,
			fieldNames,
		}
	}

	resolveModel(input: {
		mode: CoreUnitMode
		manualModelId: string | null
		sensorKind: string
		payload: Record<string, unknown>
		policy?: AutoRoutingPolicy
		anomalyDetection?: boolean
	}): ModelRoutingDecision {
		const profile = this.classifyEvent(input.sensorKind, input.payload)

		if (input.mode !== 'auto') {
			const manual = input.manualModelId
			const entry = manual ? this.registry.findById(manual) : undefined
			return {
				modelId: manual,
				label: entry?.label ?? manual ?? 'No model selected',
				reason: manual ? 'Manual core unit selection' : 'No manual model configured',
				profile,
				plannerSource: 'manual',
			}
		}

		const policyBinding = input.policy?.bindings.find(binding => binding.sensorKind === profile.sensorKind)
		if (policyBinding) {
			return {
				modelId: policyBinding.modelId,
				label: policyBinding.label,
				reason: policyBinding.reason,
				profile,
				plannerSource: 'policy',
			}
		}

		const model = this.pickBestModel(profile, input.anomalyDetection !== false)
		return {
			modelId: model?.id ?? null,
			label: model?.label ?? 'No compatible model',
			reason: model
				? `Auto-selected ${model.label} for ${profile.modality}/${profile.preferredTask} (${profile.sensorKind})`
				: `No trained model matches ${profile.modality} ${profile.preferredTask} for sensor kind ${profile.sensorKind}`,
			profile,
			plannerSource: 'rules',
		}
	}

	buildPolicyBindings(sensorKinds: string[]): AutoRoutingBinding[] {
		const bindings: AutoRoutingBinding[] = []
		for (const sensorKind of sensorKinds) {
			const samplePayload = this.samplePayloadForSensorKind(sensorKind)
			const profile = this.classifyEvent(sensorKind, samplePayload)
			const model = this.pickBestModel(profile, true)
			if (!model) continue
			bindings.push({
				sensorKind,
				modelId: model.id,
				label: model.label,
				reason: `Routes ${sensorKind} telemetry to ${model.kind} (${model.dataset})`,
			})
		}
		return bindings
	}

	buildDefaultPolicy(sensorKinds: string[]): AutoRoutingPolicy {
		const registryCount = this.registry.listModels().length
		const bindings = this.buildPolicyBindings(sensorKinds)
		let summary: string
		if (registryCount === 0) {
			summary =
				'Auto mode enabled, but the trained model registry is empty on this server. Ensure models/reports and models/checkpoints are available to the backend.'
		} else if (sensorKinds.length === 0) {
			summary =
				'Auto mode enabled. Add Stage 01 sensor sources, then save Auto mode again to generate sensor-to-model bindings.'
		} else if (bindings.length === 0) {
			summary = `Auto mode enabled, but no compatible trained models match sensor kinds: ${sensorKinds.join(', ')}.`
		} else {
			summary = `Auto mode binds ${bindings.length} sensor profile(s) to modality-safe trained models.`
		}
		return {
			generatedAt: new Date().toISOString(),
			plannerSource: 'rules',
			summary,
			bindings,
		}
	}

	getRegistryModelCount(): number {
		return this.registry.listModels().length
	}

	private inferModality(
		sensorKind: string,
		hasWeatherFields: boolean,
		hasTrafficFields: boolean,
	): ModelModality {
		if (sensorKind === 'weather' || (hasWeatherFields && !hasTrafficFields)) return 'weather'
		if (sensorKind === 'traffic' || hasTrafficFields) return 'traffic'
		return 'generic'
	}

	private inferPreferredTask(
		payload: Record<string, unknown>,
		hasTrafficFields: boolean,
		hasWeatherFields: boolean,
	): ModelTask {
		const values = Object.values(payload)
		const missingRatio =
			values.length === 0
				? 0
				: values.filter(value => value === null || value === undefined || value === '').length / values.length
		if (missingRatio >= 0.2) return 'imputation'
		if (hasTrafficFields || hasWeatherFields) return 'forecast'
		return 'anomaly'
	}

	private pickBestModel(profile: EventRoutingProfile, preferAnomaly: boolean) {
		const models = this.registry.listModels()

		if (preferAnomaly) {
			const anomalyModel = this.findBestForTask(models, profile, 'anomaly')
			if (anomalyModel) return anomalyModel
		}

		return this.findBestForTask(models, profile, profile.preferredTask)
	}

	private findBestForTask(models: ModelRegistryEntry[], profile: EventRoutingProfile, task: ModelTask) {
		const compatible = models
			.filter(model => model.modality === profile.modality)
			.filter(model => model.sensorKinds.includes(profile.sensorKind) || model.sensorKinds.includes('generic'))
			.filter(model => !model.requiresGeo || profile.hasGeo)
			.filter(model => model.tasks.includes(task) || (task !== 'anomaly' && model.tasks.includes('forecast')))
			.sort((left, right) => this.scoreModel(left, profile) - this.scoreModel(right, profile))

		return compatible[0]
	}

	private scoreModel(model: ModelRegistryEntry, profile: EventRoutingProfile) {
		let score = model.priority
		if (profile.modality === 'traffic' && model.dataset.includes('astana')) score -= 3
		if (profile.modality === 'traffic' && model.dataset.includes('traffic')) score -= 2
		if (profile.modality === 'weather' && model.dataset.includes('weather')) score -= 4
		if (profile.modality === 'weather' && model.modality !== 'weather') score += 100
		if (profile.modality === 'traffic' && model.modality === 'weather') score += 100
		if (profile.preferredTask === 'anomaly' && model.tasks.includes('anomaly')) score -= 2
		if (profile.preferredTask === 'forecast' && model.tasks.includes('forecast')) score -= 1
		if (profile.hasGeo && model.requiresGeo) score -= 1
		return score
	}

	private samplePayloadForSensorKind(sensorKind: string): Record<string, unknown> {
		switch (sensorKind) {
			case 'traffic':
				return {
					speedKmh: 42,
					trafficDensity: 68,
					latitude: 51.12,
					longitude: 71.45,
					vehicleCount: 120,
				}
			case 'weather':
				return { temperatureC: 4.2, humidityPct: 71, pressureHpa: 1012, windSpeedKmh: 12 }
			case 'air':
				return { pm25: 18, pm10: 24, humidityPct: 55, temperatureC: 6.1 }
			default:
				return { value: 1 }
		}
	}
}
