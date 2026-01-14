/**
 * Flowmatic API Documentation
 * Base URL: http://localhost:3000/api/v1
 */

// ============================================================================
// AUTH ENDPOINTS
// ============================================================================

/**
 * POST /auth/login
 * Authenticate user with email (SSO in production)
 *
 * Request:
 * {
 *   email: string
 * }
 *
 * Response (200):
 * {
 *   success: boolean
 *   user: {
 *     id: string
 *     email: string
 *     displayName: string
 *     organizationId: string
 *     role: 'admin' | 'member' | 'viewer'
 *   }
 * }
 *
 * Set-Cookie: flowmatic_session=<token>; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000
 */

/**
 * GET /auth/profile
 * Get current user profile (requires auth)
 *
 * Response (200):
 * {
 *   userId: string
 *   organizationId: string
 *   role: 'admin' | 'member' | 'viewer'
 * }
 */

/**
 * POST /auth/logout
 * Logout current user (requires auth)
 *
 * Response (200):
 * {
 *   success: boolean
 * }
 *
 * Clears: flowmatic_session cookie
 */

// ============================================================================
// INGESTION ENDPOINTS
// ============================================================================

/**
 * POST /ingestion/upload
 * Upload and process a file (requires auth)
 *
 * Request: multipart/form-data
 * - file: File (CSV/JSON)
 *
 * Response (200):
 * {
 *   success: boolean
 *   message: string
 *   data: {
 *     runId: string (UUID of the pipeline run)
 *     jobId: string (Queue job ID)
 *     fileId: string (Storage file ID)
 *     fileName: string
 *     fileSize: number (bytes)
 *     preview: {
 *       columns: string[]
 *       rowCount: number
 *       sampleRows: Record<string, any>[]
 *     }
 *   }
 * }
 *
 * Notes:
 * - File is stored in S3 with key: uploads/{organizationId}/{timestamp}_{random}_{filename}
 * - Processing happens asynchronously via RabbitMQ queue
 * - Poll /pipelines/runs/:runId to check status
 */

// ============================================================================
// PIPELINE ENDPOINTS
// ============================================================================

/**
 * GET /pipelines/runs
 * List all pipeline runs for the organization (requires auth)
 *
 * Query Parameters:
 * - limit: number (default: 50, max: 500)
 *
 * Response (200):
 * {
 *   success: boolean
 *   data: PipelineRun[]
 * }
 *
 * PipelineRun:
 * {
 *   id: string (UUID)
 *   organizationId: string
 *   status: 'queued' | 'processing' | 'completed' | 'failed'
 *   sourceFileName: string
 *   sourceFileId: string
 *   rowsIngested: number
 *   rowsCleaned: number
 *   rowsErrors: number
 *   processingTimeMs: number
 *   jobId: string | null
 *   errorMessage: string | null
 *   resultFileId: string | null
 *   resultFileSize: number | null
 *   createdAt: ISO8601 timestamp
 *   updatedAt: ISO8601 timestamp
 *   sourceFile: StorageFile | null
 *   resultFile: StorageFile | null
 * }
 */

/**
 * GET /pipelines/runs/:runId
 * Get details of a specific pipeline run (requires auth)
 *
 * Response (200):
 * {
 *   success: boolean
 *   data: PipelineRun
 * }
 *
 * Response (404): Run not found or unauthorized
 */

// ============================================================================
// STORAGE ENDPOINTS (Future)
// ============================================================================

/**
 * GET /storage/files/:fileId/download
 * Get signed download URL for a file (requires auth)
 *
 * Query Parameters:
 * - expiresIn: number (seconds, default: 3600)
 *
 * Response (200):
 * {
 *   success: boolean
 *   data: {
 *     url: string (signed S3 URL)
 *     expiresAt: ISO8601 timestamp
 *   }
 * }
 */

// ============================================================================
// TYPES
// ============================================================================

export interface AuthLoginRequest {
	email: string
	password?: string // For SSO providers
}

export interface AuthLoginResponse {
	success: boolean
	user: {
		id: string
		email: string
		displayName: string
		organizationId: string
		role: 'admin' | 'member' | 'viewer'
	}
}

export interface AuthProfileResponse {
	userId: string
	organizationId: string
	role: 'admin' | 'member' | 'viewer'
}

export interface FileUploadResponse {
	success: boolean
	message: string
	data: {
		runId: string
		jobId: string
		fileId: string
		fileName: string
		fileSize: number
		preview: {
			columns: string[]
			rowCount: number
			sampleRows: Record<string, any>[]
		}
	}
}

export interface PipelineRun {
	id: string
	organizationId: string
	status: 'queued' | 'processing' | 'completed' | 'failed'
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

export interface PipelineRunsResponse {
	success: boolean
	data: PipelineRun[]
}

export interface PipelineRunResponse {
	success: boolean
	data: PipelineRun
}

// ============================================================================
// ERROR RESPONSES
// ============================================================================

/**
 * All endpoints may return error responses:
 *
 * 400 Bad Request
 * {
 *   statusCode: 400
 *   message: string | string[]
 *   error: string
 * }
 *
 * 401 Unauthorized
 * {
 *   statusCode: 401
 *   message: string
 *   error: string
 * }
 *
 * 403 Forbidden
 * {
 *   statusCode: 403
 *   message: string
 *   error: string
 * }
 *
 * 404 Not Found
 * {
 *   statusCode: 404
 *   message: string
 *   error: string
 * }
 *
 * 500 Internal Server Error
 * {
 *   statusCode: 500
 *   message: string
 *   error: string
 * }
 */

// ============================================================================
// WEBHOOK EVENTS (Future)
// ============================================================================

/**
 * POST /webhooks/register
 * Register a webhook for pipeline events
 *
 * Request:
 * {
 *   url: string
 *   events: ('pipeline.queued' | 'pipeline.processing' | 'pipeline.completed' | 'pipeline.failed')[]
 * }
 *
 * Webhook Payload:
 * {
 *   event: string
 *   runId: string
 *   timestamp: ISO8601
 *   data: PipelineRun
 * }
 */
