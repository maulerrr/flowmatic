/**
 * Flowmatic API Client
 * Handles all communication with the backend API
 */

export interface ApiResponse<T = unknown> {
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
	createdAt: string
	organizations?: OrganizationMembership[]
}

export interface Organization {
	id: string
	name: string
	slug: string
}

export interface OrganizationMembership extends Organization {
	role: 'admin' | 'member' | 'viewer'
	isActive: boolean
	createdAt?: string
}

export interface OrganizationMember {
	id: string
	role: 'admin' | 'member' | 'viewer'
	user: {
		id: string
		email: string
		displayName: string | null
		createdAt: string
	}
}

export interface OrganizationInvitation {
	id: string
	email: string
	role: 'admin' | 'member' | 'viewer'
	token: string
	expiresAt: string
	createdAt: string
	organization: Organization
	invitedBy?: {
		id: string
		email: string
		displayName: string | null
	}
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
	summary: string | null
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

export interface SmartCityPipeline {
	id: string
	organizationId: string
	createdByUserId: string
	name: string
	description: string | null
	status: 'DRAFT' | 'ACTIVE' | 'PAUSED' | 'ERROR' | 'ARCHIVED'
	graphJson: Record<string, any>
	streamConfig: Record<string, any>
	activeModelId: string | null
	dataLakeConnectionId: string | null
	createdAt: string
	updatedAt: string
	sensorSources?: SensorSource[]
}

export interface FederatedConnectionState {
	enabled: boolean
	protocol: 'HTTP' | 'WEBSOCKET'
	endpoint: string
	projectId?: string | null
	nodeId?: string | null
	topic?: string | null
	headers?: Record<string, string>
	registerPayload?: Record<string, unknown>
	status: 'DISCONNECTED' | 'CONNECTED' | 'ERROR'
	lastConnectedAt?: string | null
	lastTestedAt?: string | null
	lastError?: string | null
	lastTestResult?: string | null
	registrationId?: string | null
	registeredAt?: string | null
	lastDeliveryAt?: string | null
	globalModelVersion?: string | null
	currentRoundId?: string | null
	rounds?: FederatedRound[]
}

export interface FederatedParticipantUpdate {
	nodeId: string
	submittedAt: string
	sampleCount: number | null
	checkpointUri: string | null
	metrics: Record<string, unknown>
	notes: string | null
}

export interface FederatedRound {
	id: string
	name: string
	status: 'ACTIVE' | 'AWAITING_UPDATES' | 'AGGREGATING' | 'COMPLETED' | 'FAILED'
	startedAt: string
	completedAt: string | null
	sampleCount: number | null
	metrics: Record<string, unknown>
	participants: FederatedParticipantUpdate[]
	aggregatedModelVersion: string | null
	aggregatedCheckpointUri: string | null
	aggregationMetrics: Record<string, unknown>
	summary: string | null
}

export interface SensorSource {
	id: string
	organizationId: string
	pipelineId: string
	name: string
	type: 'WEBSOCKET' | 'HTTP_POLLING'
	sensorKind: string
	mode: 'SIMULATED' | 'EXTERNAL'
	status: 'STOPPED' | 'RUNNING' | 'ERROR'
	endpoint: string | null
	schemaJson: Record<string, any>
	connectionConfig: Record<string, any>
	pollIntervalMs: number
	lastSeenAt: string | null
	lastError: string | null
	createdAt: string
	updatedAt: string
}

export interface SensorEvent {
	id: string
	organizationId: string
	pipelineId: string
	sourceId: string
	eventTime: string
	sensorType: string
	location: string | null
	payloadJson: Record<string, any>
	createdAt: string
	source?: Pick<SensorSource, 'id' | 'name' | 'type' | 'sensorKind' | 'status'>
}

export interface SensorSourceTestResult {
	ok: boolean
	mode: 'SIMULATED' | 'EXTERNAL'
	type: 'WEBSOCKET' | 'HTTP_POLLING'
	summary?: string
	event?: SensorEvent | Record<string, unknown> | null
	eventCount?: number
}

export interface FederatedConnectionResult {
	ok: boolean
	protocol: 'HTTP' | 'WEBSOCKET'
	endpoint: string
	summary: string
}

export interface ExportAdapterMetadata {
	type: string
	name: string
	description: string
	requiredSettings: string[]
}

export interface SmartCityStageExportResult {
	success: boolean
	adapterType: string
	fileName: string
	destination: string
	recordsExported: number
	message: string
	metadata?: Record<string, unknown>
	stage: 'raw' | 'cleaned' | 'business'
	rowCount: number
}

export interface SmartCityBackfillResult {
	pipelineId: string
	stage: 'raw' | 'cleaned' | 'business' | 'all'
	limit: number
	scannedEvents: number
	counts: {
		raw: number
		cleaned: number
		business: number
	}
	completedAt: string
	message: string
}

export interface SmartCityExportTarget {
	id: string
	organizationId: string
	pipelineId: string
	createdByUserId: string
	name: string
	stage: 'raw' | 'cleaned' | 'business'
	adapterType: string
	settingsJson: Record<string, unknown>
	saveCredentials: boolean
	isContinuous: boolean
	status: 'ACTIVE' | 'PAUSED' | 'ERROR' | 'ARCHIVED'
	cadenceSeconds: number
	lastCursorAt: string | null
	lastRunAt: string | null
	lastRunId: string | null
	lastError: string | null
	createdAt: string
	updatedAt: string
}

export interface SmartCityExportRun {
	id: string
	organizationId: string
	pipelineId: string
	targetId: string | null
	stage: 'raw' | 'cleaned' | 'business'
	adapterType: string
	status: 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED'
	rowCount: number
	recordsExported: number
	destination: string | null
	message: string | null
	errorMessage: string | null
	metadata: Record<string, unknown>
	startedAt: string | null
	finishedAt: string | null
	createdAt: string
	updatedAt: string
}

export interface DataLakeObject {
	key: string
	size: number
	lastModified: string | null
	etag: string | null
}

export interface DataLakeObjectGroup {
	stage: 'raw' | 'cleaned' | 'business'
	prefix: string
	objects: DataLakeObject[]
}

export interface SmartCityDataLakeObjectsResponse {
	stage: string
	objects: DataLakeObjectGroup[]
}

export interface CreateSmartCityPipelineInput {
	name: string
	description?: string
}

export interface UpdateSmartCityPipelineInput {
	name?: string
	description?: string
	status?: SmartCityPipeline['status']
	streamConfig?: Record<string, unknown>
	activeModelId?: string | null
	dataLakeConnectionId?: string | null
}

export interface CreateSensorSourceInput {
	name: string
	type: 'WEBSOCKET' | 'HTTP_POLLING'
	sensorKind?: string
	mode?: 'SIMULATED' | 'EXTERNAL'
	endpoint?: string
	pollIntervalMs?: number
	connectionConfig?: Record<string, unknown>
}

export interface ConnectFederatedLearningInput {
	endpoint: string
	protocol: 'HTTP' | 'WEBSOCKET'
	projectId?: string
	nodeId?: string
	topic?: string
	apiKey?: string
	headers?: Record<string, string>
	registerPayload?: Record<string, unknown>
}

export interface ConnectFederatedLearningResponse {
	pipeline: SmartCityPipeline
	connection: FederatedConnectionResult
}

export interface FederatedRoundCollection {
	registrationId: string | null
	globalModelVersion: string | null
	currentRoundId: string | null
	rounds: FederatedRound[]
}

export interface SmartCityObservability {
	windowHours: number
	eventsLast24h: number
	runningSources: number
	sourceErrors: number
	exportSuccesses: number
	exportFailures: number
	trainingRuns: number
	federatedStatus: string
	activeRoundId: string | null
	globalModelVersion: string | null
	alerts: Array<{ severity: 'info' | 'warning' | 'error'; message: string }>
	lastEventAt: string | null
	lastExportAt: string | null
	lastTrainingAt: string | null
}

export interface SmartCityHttpPollResult {
	generated: SensorEvent[]
	latest: SensorEvent[]
	nextPollMs: number | null
}

export interface CreateDataLakeInput {
	name: string
	provider?: DataLakeConnection['provider']
	bucket: string
	region?: string
	endpoint?: string
	basePrefix?: string
	accessKey?: string
	secretKey?: string
	isDefault?: boolean
	pathRules?: Record<string, unknown>
}

export interface UpdateDataLakeInput {
	name?: string
	provider?: DataLakeConnection['provider']
	bucket?: string
	region?: string
	endpoint?: string
	basePrefix?: string
	accessKey?: string
	secretKey?: string
	isDefault?: boolean
	pathRules?: Record<string, unknown>
}

export interface TrainSmartCityModelInput {
	name?: string
	pipelineId?: string
	datasetPath?: string
	modelType?: 'TRAFFIC_BASELINE' | 'ANOMALY_BASELINE'
}

export interface DataLakeConnection {
	id: string
	organizationId: string
	name: string
	provider: 'AWS_S3' | 'MINIO' | 'R2' | 'CUSTOM_S3'
	bucket: string
	region: string | null
	endpoint: string | null
	basePrefix: string
	hasAccessKey: boolean
	hasSecretKey: boolean
	isDefault: boolean
	status: 'CONNECTED' | 'DISCONNECTED' | 'ERROR'
	lastTestedAt: string | null
	lastTestStatus: string | null
	pathRules: Record<string, any>
	createdAt: string
	updatedAt: string
}

export interface SmartCityDashboard {
	pipeline: SmartCityPipeline
	sources: SensorSource[]
	events: SensorEvent[]
	dataLakes: DataLakeConnection[]
	stats: {
		sourceCount: number
		runningSources: number
		eventCount: number
		lastEventAt: string | null
	}
}

export interface ModelArtifact {
	id: string
	organizationId: string
	trainingRunId: string
	name: string
	modelType: string
	version: string
	status: 'LOCAL_ONLY' | 'PROMOTED' | 'DEPLOYED' | 'ARCHIVED'
	localPath: string
	s3Uri: string | null
	featureSpec: Record<string, any>
	metricsJson: Record<string, any>
	createdAt: string
	updatedAt: string
}

export interface ModelTrainingRun {
	id: string
	organizationId: string
	pipelineId: string | null
	createdByUserId: string
	name: string
	datasetPath: string
	modelType: string
	status: 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED'
	localArtifactPath: string | null
	datasetProfile: Record<string, any>
	featureSpec: Record<string, any>
	metricsJson: Record<string, any>
	logsJson: string[]
	errorMessage: string | null
	startedAt: string | null
	finishedAt: string | null
	createdAt: string
	updatedAt: string
	artifacts?: ModelArtifact[]
}

export interface ResearchModel {
	id: string
	run: string
	name: string
	kind: string
	dataset: string
	metrics: Record<string, any>
	production: Record<string, any>
	localPath: string
	hasTorchScript: boolean
	hasSafetensors: boolean
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
			body?: unknown
			query?: Record<string, unknown>
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
			credentials: 'include',
		}

		if (options?.body !== undefined) {
			fetchOptions.headers = {
				'Content-Type': 'application/json',
			}
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
	async login(email: string, password: string): Promise<ApiResponse<{ user: User }>> {
		return this.request('POST', '/auth/login', {
			body: { email, password },
		})
	}

	async register(input: {
		email: string
		password: string
		displayName: string
		organizationName?: string
	}): Promise<ApiResponse<{ user: User }>> {
		return this.request('POST', '/auth/register', { body: input })
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

	async updateProfile(displayName: string): Promise<ApiResponse<User>> {
		return this.request('PATCH', '/auth/profile', { body: { displayName } })
	}

	async listOrganizations(): Promise<ApiResponse<OrganizationMembership[]>> {
		return this.request('GET', '/auth/organizations')
	}

	async createOrganization(name: string): Promise<ApiResponse<Organization>> {
		return this.request('POST', '/auth/organizations', { body: { name } })
	}

	async updateOrganization(
		organizationId: string,
		name: string,
	): Promise<ApiResponse<Organization>> {
		return this.request('PATCH', `/auth/organizations/${organizationId}`, { body: { name } })
	}

	async switchOrganization(organizationId: string): Promise<ApiResponse<OrganizationMembership>> {
		return this.request('POST', '/auth/switch-organization', { body: { organizationId } })
	}

	async deleteOrganization(
		organizationId: string,
		confirmationName: string,
	): Promise<ApiResponse<{ deleted: boolean; nextOrganizationId: string | null }>> {
		return this.request('DELETE', `/auth/organizations/${organizationId}`, {
			body: { confirmationName },
		})
	}

	async listOrganizationMembers(
		organizationId: string,
	): Promise<ApiResponse<OrganizationMember[]>> {
		return this.request('GET', `/auth/organizations/${organizationId}/members`)
	}

	async inviteMember(
		organizationId: string,
		email: string,
		role: 'admin' | 'member' | 'viewer' = 'member',
	): Promise<ApiResponse<OrganizationInvitation>> {
		return this.request('POST', `/auth/organizations/${organizationId}/invitations`, {
			body: { email, role },
		})
	}

	async listInvitations(): Promise<ApiResponse<OrganizationInvitation[]>> {
		return this.request('GET', '/auth/invitations')
	}

	async acceptInvitation(invitationId: string): Promise<ApiResponse<Organization>> {
		return this.request('POST', `/auth/invitations/${invitationId}/accept`)
	}

	async declineInvitation(invitationId: string): Promise<ApiResponse> {
		return this.request('POST', `/auth/invitations/${invitationId}/decline`)
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

	// Smart city real-time pipeline endpoints
	async getSimulatorPresets(): Promise<ApiResponse<Record<string, unknown>>> {
		return this.request('GET', '/smart-city/simulator/presets')
	}

	async listSmartCityPipelines(): Promise<ApiResponse<SmartCityPipeline[]>> {
		return this.request('GET', '/smart-city/pipelines')
	}

	async createSmartCityPipeline(
		input: CreateSmartCityPipelineInput,
	): Promise<ApiResponse<SmartCityPipeline>> {
		return this.request('POST', '/smart-city/pipelines', { body: input })
	}

	async updateSmartCityPipeline(
		id: string,
		input: UpdateSmartCityPipelineInput,
	): Promise<ApiResponse<SmartCityPipeline>> {
		return this.request('PATCH', `/smart-city/pipelines/${id}`, { body: input })
	}

	async deleteSmartCityPipeline(id: string): Promise<ApiResponse<{ deleted: boolean }>> {
		return this.request('DELETE', `/smart-city/pipelines/${id}`)
	}

	async getSmartCityDashboard(id: string): Promise<ApiResponse<SmartCityDashboard>> {
		return this.request('GET', `/smart-city/pipelines/${id}/dashboard`)
	}

	async getSmartCityN8nWorkflow(id: string): Promise<ApiResponse<any>> {
		return this.request('GET', `/smart-city/pipelines/${id}/n8n-workflow`)
	}

	async createSensorSource(
		pipelineId: string,
		input: CreateSensorSourceInput,
	): Promise<ApiResponse<SensorSource>> {
		return this.request('POST', `/smart-city/pipelines/${pipelineId}/sources`, { body: input })
	}

	async deleteSensorSource(id: string): Promise<ApiResponse<{ deleted: boolean }>> {
		return this.request('DELETE', `/smart-city/sources/${id}`)
	}

	async startSensorSource(id: string): Promise<ApiResponse<SensorSource>> {
		return this.request('POST', `/smart-city/sources/${id}/start`)
	}

	async stopSensorSource(id: string): Promise<ApiResponse<SensorSource>> {
		return this.request('POST', `/smart-city/sources/${id}/stop`)
	}

	async connectFederatedLearning(
		pipelineId: string,
		input: ConnectFederatedLearningInput,
	): Promise<ApiResponse<ConnectFederatedLearningResponse>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/federated/connect', {
			body: input,
		})
	}

	async testFederatedLearning(
		pipelineId: string,
	): Promise<ApiResponse<FederatedConnectionResult>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/federated/test')
	}

	async disconnectFederatedLearning(
		pipelineId: string,
	): Promise<ApiResponse<SmartCityPipeline>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/federated/disconnect')
	}

	async listFederatedRounds(
		pipelineId: string,
	): Promise<ApiResponse<FederatedRoundCollection>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/federated/rounds')
	}

	async getSmartCityObservability(
		pipelineId: string,
	): Promise<ApiResponse<SmartCityObservability>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/observability')
	}

	async startFederatedRound(
		pipelineId: string,
		input: { name?: string; sampleCount?: number; metrics?: Record<string, unknown> },
	): Promise<ApiResponse<{ round: FederatedRound; pipeline: SmartCityPipeline }>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/federated/rounds', {
			body: input,
		})
	}

	async submitFederatedUpdate(
		pipelineId: string,
		roundId: string,
		input: {
			checkpointUri?: string
			sampleCount?: number
			metrics?: Record<string, unknown>
			notes?: string
		},
	): Promise<ApiResponse<{ round: FederatedRound; pipeline: SmartCityPipeline }>> {
		return this.request(
			'POST',
			'/smart-city/pipelines/' + pipelineId + '/federated/rounds/' + roundId + '/submit',
			{ body: input },
		)
	}

	async aggregateFederatedRound(
		pipelineId: string,
		roundId: string,
		input: {
			globalModelVersion?: string
			checkpointUri?: string
			metrics?: Record<string, unknown>
			summary?: string
		},
	): Promise<ApiResponse<{ round: FederatedRound; pipeline: SmartCityPipeline }>> {
		return this.request(
			'POST',
			'/smart-city/pipelines/' + pipelineId + '/federated/rounds/' + roundId + '/aggregate',
			{ body: input },
		)
	}

	async syncFederatedGlobalModel(
		pipelineId: string,
		input: { includeRounds?: boolean } = {},
	): Promise<
		ApiResponse<{
			pipeline: SmartCityPipeline
			globalModelVersion: string | null
			currentRoundId: string | null
			summary: string
			rounds: FederatedRound[]
		}>
	> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/federated/sync-global-model', {
			body: input,
		})
	}

	async testSensorSource(id: string): Promise<ApiResponse<SensorSourceTestResult>> {
		return this.request('POST', '/smart-city/sources/' + id + '/test')
	}

	async listSensorEvents(
		pipelineId: string,
		limit: number = 50,
	): Promise<ApiResponse<SensorEvent[]>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/events', {
			query: { limit },
		})
	}

	async pollSmartCityHttpStream(
		pipelineId: string,
	): Promise<ApiResponse<SmartCityHttpPollResult>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/http-poll')
	}

	async listDataLakes(): Promise<ApiResponse<DataLakeConnection[]>> {
		return this.request('GET', '/smart-city/data-lakes')
	}

	async createDataLake(input: CreateDataLakeInput): Promise<ApiResponse<DataLakeConnection>> {
		return this.request('POST', '/smart-city/data-lakes', { body: input })
	}

	async updateDataLake(
		id: string,
		input: UpdateDataLakeInput,
	): Promise<ApiResponse<DataLakeConnection>> {
		return this.request('PATCH', '/smart-city/data-lakes/' + id, { body: input })
	}

	async deleteDataLake(id: string): Promise<ApiResponse<{ deleted: boolean }>> {
		return this.request('DELETE', '/smart-city/data-lakes/' + id)
	}

	async disconnectDataLake(id: string): Promise<ApiResponse<DataLakeConnection>> {
		return this.request('POST', '/smart-city/data-lakes/' + id + '/disconnect')
	}

	async testDataLake(id: string): Promise<ApiResponse<DataLakeConnection>> {
		return this.request('POST', '/smart-city/data-lakes/' + id + '/test')
	}

	async listModelArtifacts(): Promise<ApiResponse<ModelArtifact[]>> {
		return this.request('GET', '/smart-city/models')
	}

	async listResearchModels(): Promise<ApiResponse<ResearchModel[]>> {
		return this.request('GET', '/smart-city/research-models')
	}

	async listModelTrainingRuns(): Promise<ApiResponse<ModelTrainingRun[]>> {
		return this.request('GET', '/smart-city/model-training-runs')
	}

	async trainSmartCityModel(
		input: TrainSmartCityModelInput,
	): Promise<ApiResponse<ModelTrainingRun>> {
		return this.request('POST', '/smart-city/models/train', { body: input })
	}

	async promoteSmartCityModel(id: string): Promise<ApiResponse<ModelArtifact>> {
		return this.request('POST', '/smart-city/models/' + id + '/promote')
	}

	async deploySmartCityModel(
		pipelineId: string,
		modelId: string,
	): Promise<ApiResponse<SmartCityPipeline>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/models/' + modelId + '/deploy')
	}

	async deployResearchModel(
		pipelineId: string,
		run: string,
	): Promise<ApiResponse<SmartCityPipeline>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/research-models/' + run + '/deploy')
	}

	async testProcessing(
		pipelineId: string,
		payload: Record<string, unknown> = {},
	): Promise<ApiResponse<any>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/test-processing', {
			body: { payload },
		})
	}

	async exportSmartCityPipelineStage(
		pipelineId: string,
		input: {
			adapterType: string
			stage: 'raw' | 'cleaned' | 'business'
			settings?: Record<string, unknown>
			saveCredentials?: boolean
			limit?: number
		},
	): Promise<ApiResponse<SmartCityStageExportResult>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/exports', {
			body: input,
		})
	}

	async backfillSmartCityPipeline(
		pipelineId: string,
		input: {
			stage?: 'raw' | 'cleaned' | 'business' | 'all'
			limit?: number
		},
	): Promise<ApiResponse<SmartCityBackfillResult>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/backfill', {
			body: input,
		})
	}

	async listSmartCityExportTargets(
		pipelineId: string,
	): Promise<ApiResponse<SmartCityExportTarget[]>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/export-targets')
	}

	async listSmartCityExportRuns(
		pipelineId: string,
	): Promise<ApiResponse<SmartCityExportRun[]>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/export-runs')
	}

	async createSmartCityExportTarget(
		pipelineId: string,
		input: {
			name: string
			stage: 'raw' | 'cleaned' | 'business'
			adapterType: string
			settings?: Record<string, unknown>
			saveCredentials?: boolean
			isContinuous?: boolean
			cadenceSeconds?: number
		},
	): Promise<ApiResponse<SmartCityExportTarget>> {
		return this.request('POST', '/smart-city/pipelines/' + pipelineId + '/export-targets', {
			body: input,
		})
	}

	async updateSmartCityExportTarget(
		targetId: string,
		input: Partial<{
			name: string
			stage: 'raw' | 'cleaned' | 'business'
			adapterType: string
			settings: Record<string, unknown>
			saveCredentials: boolean
			isContinuous: boolean
			cadenceSeconds: number
		}>,
	): Promise<ApiResponse<SmartCityExportTarget>> {
		return this.request('PATCH', '/smart-city/export-targets/' + targetId, {
			body: input,
		})
	}

	async runSmartCityExportTarget(
		targetId: string,
	): Promise<ApiResponse<{ queued: boolean; targetId: string }>> {
		return this.request('POST', '/smart-city/export-targets/' + targetId + '/run')
	}

	async listSmartCityDataLakeObjects(
		pipelineId: string,
		stage?: 'raw' | 'cleaned' | 'business',
	): Promise<ApiResponse<SmartCityDataLakeObjectsResponse>> {
		return this.request('GET', '/smart-city/pipelines/' + pipelineId + '/data-lake-objects', {
			query: { stage },
		})
	}

	// Export endpoints
	async getExportAdapters(): Promise<ApiResponse<ExportAdapterMetadata[]>> {
		return this.request('GET', '/exports/adapters')
	}

	async previewPipelineData(
		runId: string,
		page: number = 1,
		pageSize: number = 25,
	): Promise<ApiResponse<any>> {
		return this.request('GET', '/exports/runs/:runId/preview'.replace(':runId', runId), {
			query: { page, pageSize },
		})
	}

	async exportPipelineRun(
		runId: string,
		adapterType: string,
		settings: Record<string, any>,
		saveCredentials: boolean = false,
	): Promise<ApiResponse<any>> {
		return this.request('POST', '/exports/runs/' + runId + '/export', {
			body: { adapterType, settings, saveCredentials },
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
		return this.request('GET', '/exports/runs/' + runId + '/history')
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
