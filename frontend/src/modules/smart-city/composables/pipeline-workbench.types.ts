export type PipelineStage = 'sources' | 'processing' | 'lake' | 'federated'
export type ExportStage = 'raw' | 'cleaned' | 'business'
export type ExportAdapterKind = 'json' | 'csv' | 'postgres' | 'mongodb' | 'huggingface'

export interface N8nWorkflowSnapshot {
	nodes?: Array<Record<string, unknown>>
	[key: string]: unknown
}

export interface ProcessingTestResult {
	error?: string
	output?: {
		result?: unknown
		[key: string]: unknown
	}
	[key: string]: unknown
}
