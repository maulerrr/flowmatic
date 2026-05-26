import { CopilotVizSpec } from './pipeline-copilot.types'

export type InsightDepth = 'quick' | 'standard' | 'deep'
export type InsightFocus = 'all' | 'ops' | 'quality' | 'geo' | 'exports'
export type InsightTrigger = 'scheduled' | 'manual'
export type DataFieldKind = 'timestamp' | 'latitude' | 'longitude' | 'numeric' | 'category' | 'text' | 'id' | 'unknown'

export interface PipelineInsightConfig {
	intervalMinutes: number
	depth: InsightDepth
	focus: InsightFocus
	enabled: boolean
	lastRunAt: string | null
	nextRunAt: string | null
}

export interface DataFieldProfile {
	name: string
	kind: DataFieldKind
	sampleValues: Array<string | number | null>
	nonNullPct: number
	uniqueCount: number
	min?: number
	max?: number
}

export interface PipelineDataProfile {
	generatedAt: string
	sampleSize: number
	rowCount: number
	fields: DataFieldProfile[]
	domains: Array<'geospatial' | 'timeseries' | 'categorical' | 'numeric' | 'text'>
	hasGeospatial: boolean
	hasTimeseries: boolean
	geoBounds?: {
		minLat: number
		maxLat: number
		minLng: number
		maxLng: number
		pointCount: number
	}
	latField?: string
	lngField?: string
	timeField?: string
	sensorTypes: string[]
	sourceCount: number
}

export interface InsightFinding {
	id: string
	category: 'pattern' | 'correlation' | 'anomaly' | 'quality' | 'ops' | 'geo'
	severity: 'info' | 'warning' | 'critical'
	title: string
	summary: string
	evidence?: Record<string, unknown>
	recommendedViz?: string
}

export interface InsightAction {
	id: string
	label: string
	description: string
	priority: 'low' | 'medium' | 'high'
	kind: 'ops' | 'quality' | 'export' | 'source' | 'model'
}

export interface PipelineAnalysisFindings {
	generatedAt: string
	findings: InsightFinding[]
	correlations: Array<{ fieldA: string; fieldB: string; coefficient: number }>
	geoHotspots: Array<{ lat: number; lng: number; weight: number; label: string }>
	hourlyPattern?: Array<{ hour: string; count: number }>
	featureEngineering: Array<{ name: string; formula: string; purpose: string }>
}

export interface PipelineInsightPlan {
	narrative: string
	headline: string
	findings: InsightFinding[]
	visualizationIds: string[]
	actions: InsightAction[]
	plannerSource: 'llm' | 'rules'
}

export interface PipelineInsightRunSummary {
	id: string
	pipelineId: string
	trigger: InsightTrigger
	status: string
	depth: InsightDepth
	focus: InsightFocus
	headline: string
	narrative: string
	profile: PipelineDataProfile
	findings: InsightFinding[]
	actions: InsightAction[]
	visualizations: CopilotVizSpec[]
	plannerSource: 'llm' | 'rules'
	startedAt: string
	finishedAt: string | null
	createdAt: string
	errorMessage?: string | null
}

export const INSIGHT_VIZ_CATALOG = [
	{ id: 'event_volume', type: 'area' as const, label: 'Event volume timeline' },
	{ id: 'export_health', type: 'donut' as const, label: 'Export health' },
	{ id: 'data_funnel', type: 'funnel' as const, label: 'Medallion funnel' },
	{ id: 'source_health', type: 'bar' as const, label: 'Source health' },
	{ id: 'quality_kpis', type: 'kpi' as const, label: 'Quality KPIs' },
	{ id: 'geo_map', type: 'map' as const, label: 'Geospatial map' },
	{ id: 'geo_heatmap', type: 'heatmap' as const, label: 'Spatial heatmap' },
	{ id: 'metric_scatter', type: 'scatter' as const, label: 'Metric correlation scatter' },
	{ id: 'multi_timeline', type: 'timeline' as const, label: 'Multi-series timeline' },
	{ id: 'export_adapters', type: 'bar' as const, label: 'Export adapters' },
] as const

export type InsightVizId = (typeof INSIGHT_VIZ_CATALOG)[number]['id']
