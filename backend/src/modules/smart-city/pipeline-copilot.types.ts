export type CopilotVizType =
	| 'area'
	| 'bar'
	| 'donut'
	| 'table'
	| 'kpi'
	| 'funnel'
	| 'scatter'
	| 'map'
	| 'heatmap'
	| 'timeline'

export interface CopilotVizSeries {
	name: string
	values: number[]
	color?: string
}

export interface CopilotVizSpec {
	id: string
	title: string
	subtitle?: string
	type: CopilotVizType
	labels: string[]
	series: CopilotVizSeries[]
	rows?: Array<Record<string, string | number | null>>
	kpis?: Array<{ label: string; value: string | number; hint?: string }>
	meta?: Record<string, unknown>
}

export interface CopilotChip {
	id: string
	label: string
	category: 'overview' | 'sources' | 'core' | 'exports' | 'data'
	priority: number
}

export interface CopilotVizCommand {
	id: string
	slash: string
	label: string
	description: string
	category: CopilotChip['category']
}

export interface PipelineCopilotSourceSummary {
	id: string
	name: string
	type: string
	sensorKind: string
	mode: string
	status: string
	lastSeenAt: string | null
	lastError: string | null
}

export interface PipelineCopilotExportTargetSummary {
	id: string
	name: string
	stage: string
	adapterType: string
	status: string
	isContinuous: boolean
	cadenceSeconds: number
	lastRunAt: string | null
	lastError: string | null
	recordsExportedTotal: number
}

export interface PipelineCopilotExportRunSummary {
	id: string
	stage: string
	adapterType: string
	status: string
	rowCount: number
	recordsExported: number
	destination: string | null
	createdAt: string
	errorMessage: string | null
}

export interface PipelineCopilotContext {
	generatedAt: string
	windowHours: number
	pipeline: {
		id: string
		name: string
		status: string
		description: string | null
		isLive: boolean
	}
	runtime: {
		wsConnected: boolean
		runningSources: number
		recentEventsBuffered: number
		exportCadenceSeconds: number | null
	}
	sources: {
		total: number
		running: number
		errors: number
		items: PipelineCopilotSourceSummary[]
	}
	coreUnit: {
		activeModelId: string | null
		activeModelLabel: string
		coreUnitMode: 'manual' | 'auto'
		autoRoutingSummary?: string | null
		lastAutoResolution?: {
			modelId: string
			label: string
			reason: string
			sensorKind: string
			at: string
		} | null
		autoCleaning: boolean
		anomalyDetection: boolean
		schemaValidation: boolean
	}
	exports: {
		targetCount: number
		continuousCount: number
		errorTargetCount: number
		successRuns24h: number
		failedRuns24h: number
		totalRowsExported24h: number
		targets: PipelineCopilotExportTargetSummary[]
		recentRuns: PipelineCopilotExportRunSummary[]
	}
	dataFlow: {
		totalEvents: number
		eventsLast24h: number
		lastEventAt: string | null
		stageCounts: { raw: number; cleaned: number; business: number }
		hourlyEvents: Array<{ hour: string; count: number }>
		hourlyExportRows: Array<{ hour: string; rows: number }>
		qualitySample?: {
			missingPct: number
			duplicatePct: number
			outlierPct: number
		}
	}
	summaryText: string
	llmEnabled: boolean
	chips: CopilotChip[]
	vizCommands: CopilotVizCommand[]
	primaryVisualizations: CopilotVizSpec[]
}

export interface PipelineCopilotChatTurn {
	id: string
	role: 'assistant' | 'user'
	text: string
	chipId?: string
	answerSource?: 'llm' | 'rules' | 'system'
	visualization?: CopilotVizSpec
}

export interface PipelineCopilotChatResponse {
	context: PipelineCopilotContext
	turn: PipelineCopilotChatTurn
}
