/**
 * Flowmatic API Client
 * Handles all communication with the backend API
 */

export interface ApiResponse<T = any> {
	success: boolean
	message?: string
	data?: T
	error?: string
}

export interface User {
	id: string
	email: string
	displayName: string
	organizationId: string
	role: 'admin' | 'member' | 'viewer'
}

export interface Organization {
	id: string
	name: string
	slug: string
}

export interface AuthContext {
	userId: string
	organizationId: string
	role: 'admin' | 'member' | 'viewer'
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
	sourceFile?: {
		id: string
		fileName: string
		fileSize: number
		s3Key: string
	}
	resultFile?: {
		id: string
		fileName: string
		fileSize: number
		s3Key: string
	}
}

export interface PipelineRunPreview {
	runId: string
	fileName: string
	fileSize: number
	status: string
	stats: {
		rowsIngested: number
		rowsCleaned: number
		rowsErrors: number
		processingTimeMs: number
	}
	createdAt: string
	updatedAt: string
}

export interface FileUploadResponse extends ApiResponse {
	data?: {
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

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000'
const API_VERSION = '/api/v1'

export class ApiClient {
	private baseUrl = `${API_BASE_URL}${API_VERSION}`

	private async request<T>(
		method: string,
		path: string,
		options?: {
			body?: any
			query?: Record<string, any>
		},
	): Promise<T> {
		const url = new URL(`${this.baseUrl}${path}`)

		if (options?.query) {
			Object.entries(options.query).forEach(([key, value]) => {
				if (value !== null && value !== undefined) {
					url.searchParams.append(key, String(value))
				}
			})
		}

		const fetchOptions: RequestInit = {
			method,
			headers: {
				'Content-Type': 'application/json',
			},
			credentials: 'include',
		}

		if (options?.body) {
			fetchOptions.body = JSON.stringify(options.body)
		}

		const response = await fetch(url.toString(), fetchOptions)

		if (!response.ok) {
			const error = await response.json().catch(() => ({}))
			throw new Error(error.message || `API Error: ${response.status}`)
		}

		return response.json()
	}

	private async requestFormData<T>(method: string, path: string, formData: FormData): Promise<T> {
		const url = new URL(`${this.baseUrl}${path}`)

		const response = await fetch(url.toString(), {
			method,
			body: formData,
			credentials: 'include',
		})

		if (!response.ok) {
			const error = await response.json().catch(() => ({}))
			throw new Error(error.message || `API Error: ${response.status}`)
		}

		return response.json()
	}

	// Auth endpoints
	async login(email: string): Promise<ApiResponse<{ user: User }>> {
		return this.request('POST', '/auth/login', {
			body: { email },
		})
	}

	async getProfile(): Promise<ApiResponse<User>> {
		return this.request<ApiResponse<User>>('GET', '/auth/profile')
	}

	async changePassword(password: string): Promise<ApiResponse<{ message: string }>> {
		return this.request('POST', '/auth/change-password', {
			body: { password },
		})
	}

	async deleteAccount(): Promise<ApiResponse<{ message: string }>> {
		return this.request('POST', '/auth/delete-account')
	}

	async logout(): Promise<ApiResponse> {
		return this.request('POST', '/auth/logout')
	}

	// File upload
	async uploadFile(file: File): Promise<FileUploadResponse> {
		const formData = new FormData()
		formData.append('file', file)
		return this.requestFormData('POST', '/ingestion/upload', formData)
	}

	// Pipeline endpoints
	async listPipelineRuns(
		limit: number = 50,
		offset: number = 0,
		status?: string,
	): Promise<ApiResponse<PipelineRun[]>> {
		return this.request('GET', '/pipelines/runs', {
			query: { limit, offset, status },
		})
	}

	async getPipelineRun(runId: string): Promise<ApiResponse<PipelineRun>> {
		return this.request('GET', `/pipelines/runs/${runId}`)
	}

	async getRunPreview(runId: string): Promise<ApiResponse<PipelineRunPreview>> {
		return this.request('GET', `/pipelines/runs/${runId}/preview`)
	}

	// Analytics endpoints
	async getAnalyticsSummary(): Promise<ApiResponse<any>> {
		return this.request('GET', '/pipelines/analytics/summary')
	}

	async getAnalyticsCharts(period: string = '7d'): Promise<ApiResponse<any>> {
		return this.request('GET', '/pipelines/analytics/charts', {
			query: { period },
		})
	}

	async deleteRun(runId: string): Promise<ApiResponse> {
		return this.request('DELETE', `/pipelines/runs/${runId}`)
	}

	async cleanupOldRuns(
		daysOld: number = 30,
		statuses: string[] = ['failed', 'completed'],
	): Promise<ApiResponse<{ count: number }>> {
		return this.request('POST', '/pipelines/cleanup', {
			query: {
				daysOld,
				statuses: statuses.join(','),
			},
		})
	}

	// Export endpoints
	async getExportAdapters(): Promise<ApiResponse<any[]>> {
		return this.request('GET', '/exports/adapters')
	}

	async previewPipelineData(runId: string, limit: number = 10): Promise<ApiResponse<any>> {
		return this.request('GET', '/exports/runs/:runId/preview'.replace(':runId', runId), {
			query: { limit: String(limit) },
		})
	}

	async exportPipelineRun(
		runId: string,
		adapterType: string,
		settings: Record<string, any>,
	): Promise<ApiResponse<any>> {
		return this.request('POST', `/exports/runs/${runId}/export`, {
			body: { adapterType, settings },
		})
	}

	async validateExportConfig(
		adapterType: string,
		settings: Record<string, any>,
	): Promise<ApiResponse<any>> {
		return this.request('POST', '/exports/validate', {
			body: { adapterType, settings },
		})
	}

	async getExportHistory(runId: string): Promise<ApiResponse<any[]>> {
		return this.request('GET', `/exports/runs/${runId}/history`)
	}

	// Polling utility
	async pollPipelineRun(
		runId: string,
		maxAttempts: number = 120,
		interval: number = 1000,
		onProgress?: (run: PipelineRun) => void,
	): Promise<PipelineRun> {
		let attempts = 0

		while (attempts < maxAttempts) {
			const response = await this.getPipelineRun(runId)

			if (!response.data) {
				throw new Error('Pipeline run not found')
			}

			onProgress?.(response.data)

			if (response.data.status === 'completed' || response.data.status === 'failed') {
				return response.data
			}

			await new Promise(resolve => setTimeout(resolve, interval))
			attempts++
		}

		throw new Error('Pipeline processing timeout')
	}
}

export const apiClient = new ApiClient()
