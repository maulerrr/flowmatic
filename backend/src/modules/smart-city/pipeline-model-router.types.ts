export type CoreUnitMode = 'manual' | 'auto'

export type ModelModality = 'traffic' | 'weather' | 'energy' | 'generic'

export type ModelTask = 'forecast' | 'anomaly' | 'imputation' | 'classification' | 'repair'

export interface ModelRegistryEntry {
	id: string
	label: string
	kind: string
	dataset: string
	modality: ModelModality
	tasks: ModelTask[]
	sensorKinds: string[]
	requiresGeo?: boolean
	priority: number
	source: 'research' | 'manifest' | 'production'
	production?: boolean
	productionSlot?: string
	run?: string
	repoId?: string
}

export interface EventRoutingProfile {
	sensorKind: string
	modality: ModelModality
	hasGeo: boolean
	hasWeatherFields: boolean
	hasTrafficFields: boolean
	preferredTask: ModelTask
	fieldNames: string[]
}

export interface ModelRoutingDecision {
	modelId: string | null
	label: string
	reason: string
	profile: EventRoutingProfile
	plannerSource: 'policy' | 'rules' | 'manual'
}

export interface AutoRoutingBinding {
	sensorKind: string
	modelId: string
	label: string
	reason: string
}

export interface AutoRoutingPolicy {
	generatedAt: string
	plannerSource: 'llm' | 'rules'
	summary: string
	bindings: AutoRoutingBinding[]
}

export interface CoreUnitStreamConfig {
	coreUnitMode?: CoreUnitMode
	autoRoutingPolicy?: AutoRoutingPolicy
	lastAutoResolution?: {
		modelId: string
		label: string
		reason: string
		sensorKind: string
		at: string
	}
}
