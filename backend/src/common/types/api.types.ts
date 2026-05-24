// API Response Types
export interface ApiResponse<T = unknown> {
	success: boolean
	message?: string
	data?: T
	error?: string
}

// Auth Types
export interface User {
	id: string
	email: string
	displayName: string
	organizationId: string
	role: 'admin' | 'member' | 'viewer'
}

export interface AuthLoginRequest {
	email: string
	password?: string
}

export interface AuthLoginResponse extends ApiResponse {
	user?: User
}

export interface AuthProfileResponse extends ApiResponse {
	data?: User
}

// File Upload & Ingestion Types
export interface FilePreview {
	columns: string[]
	rowCount: number
	sampleRows: Record<string, unknown>[]
}

export interface FileUploadResponse extends ApiResponse {
	data?: {
		runId: string
		jobId: string
		fileId: string
		fileName: string
		fileSize: number
		preview: FilePreview
	}
}

// Pipeline Types
export type PipelineStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface StorageFile {
	id: string
	organizationId: string
	fileName: string
	fileSize: number
	mimeType: string
	s3Key: string
	createdAt: string
	updatedAt: string
}

export interface PipelineRun {
	id: string
	organizationId: string
	status: PipelineStatus
	sourceFileName: string
	sourceFileId: string
	rowsIngested: number
	rowsCleaned: number
	rowsErrors: number
	processingTimeMs: number
	jobId: string | null
	errorMessage: string | null
	resultFileId: string | null
	resultFileSize: number | null
	createdAt: string
	updatedAt: string
	sourceFile: StorageFile | null
	resultFile: StorageFile | null
}

export interface PipelineRunsResponse extends ApiResponse {
	data?: PipelineRun[]
}

export interface PipelineRunResponse extends ApiResponse {
	data?: PipelineRun
}

// Error Response Type
export interface ErrorResponse {
	statusCode: number
	message: string | string[]
	error: string
}

// Frontend State Types
export interface UploadState {
	file: File | null
	isLoading: boolean
	progress: number
	status: 'idle' | 'uploading' | 'success' | 'error'
	error: string | null
	result: FileUploadResponse | null
}

export interface PipelineListState {
	runs: PipelineRun[]
	isLoading: boolean
	error: string | null
	lastUpdated: Date | null
}

export interface PipelineDetailState {
	run: PipelineRun | null
	isLoading: boolean
	error: string | null
	isPolling: boolean
	pollInterval: NodeJS.Timer | null
}
