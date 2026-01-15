export interface PipelineJobData {
	runId: string
	organizationId: string
	sourceFileId: string
	sourceFileName: string
}

export interface PipelineJobResult {
	runId: string
	rowsIngested: number
	rowsCleaned: number
	rowsErrors: number
	processingTimeMs: number
	resultFileId?: string
	resultFileSize?: number
	errorMessage?: string
}

export interface PipelineSummary {
	overview: string
	scores: {
		initial: number
		final: number
	}
	insights: string[]
	recommendation: string | null
}
