import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { toast } from 'vue-sonner'
import {
	apiClient,
	type SmartCityBackfillResult,
	type SmartCityObservability,
	type DataLakeConnection,
	type DataLakeObjectGroup,
	type ExportAdapterMetadata,
	type FederatedRound,
	type SmartCityExportRun,
	type SmartCityExportPreview,
	type SmartCityExportTarget,
	type HuggingFaceIntegrationStatus,
	type HuggingFaceModelCatalogPage,
	type HuggingFaceModelSummary,
	type ModelArtifact,
	type ModelTrainingRun,
	type ResearchModel,
	type SensorEvent,
	type SensorSource,
	type SensorSourceTestResult,
	type SmartCityPipeline,
} from '@/api/client'
import { formatBytes, formatDateTime, sourceIcon } from './pipeline-display.helpers'
import {
	type ExportAdapterKind,
	type ExportStage,
	applyAdapterDefaults,
	applyExportSettingsToForm,
	buildExportSettings,
	buildExportSettingsPreview,
	buildPipelineFlowHint,
	syncStageDependentExportFields,
	validateExportTargetForm,
} from './pipeline-export.helpers'
import type { N8nWorkflowSnapshot, ProcessingTestResult } from './pipeline-workbench.types'
import { replaceInList } from './pipeline-entity.helpers'
import { createPipelineLiveStream } from './pipeline-runtime.helpers'
import { usePipelineDataLake } from './usePipelineDataLake'
import { usePipelineFederated } from './usePipelineFederated'

export type { PipelineStage, ExportStage, ExportAdapterKind } from './pipeline-workbench.types'
export type { N8nWorkflowSnapshot, ProcessingTestResult } from './pipeline-workbench.types'

type AutoRoutingBinding = {
	sensorKind: string
	modelId: string
	label: string
	reason?: string
}

type AutoRoutingPolicySnapshot = {
	summary?: string
	bindings?: AutoRoutingBinding[]
	plannerSource?: string
	generatedAt?: string
}

export function useSmartCityPipeline() {
	const route = useRoute()
	const viewMode = ref<'dashboard' | 'workflow'>('dashboard')
	const activeStage = ref<'sources' | 'processing' | 'lake' | 'federated'>('sources')
	const loading = ref(true)
	const saving = ref(false)
	const autoRoutingPhase = ref<'idle' | 'saving' | 'matching'>('idle')
	const pipelines = ref<SmartCityPipeline[]>([])
	const selectedPipelineId = ref('')
	const sources = ref<SensorSource[]>([])
	const events = ref<SensorEvent[]>([])
	const dataLakes = ref<DataLakeConnection[]>([])
	const exportAdapters = ref<ExportAdapterMetadata[]>([])
	const exportTargets = ref<SmartCityExportTarget[]>([])
	const exportRuns = ref<SmartCityExportRun[]>([])
	const exportPreview = ref<SmartCityExportPreview | null>(null)
	const exportPreviewStage = ref<ExportStage>('cleaned')
	const dataLakeObjects = ref<DataLakeObjectGroup[]>([])
	const latestBackfill = ref<SmartCityBackfillResult | null>(null)
	const federatedRounds = ref<FederatedRound[]>([])
	const observability = ref<SmartCityObservability | null>(null)
	const models = ref<ModelArtifact[]>([])
	const trainingRuns = ref<ModelTrainingRun[]>([])
	const researchModels = ref<ResearchModel[]>([])
	const latestProcessingResult = ref<ProcessingTestResult | null>(null)
	const processingError = ref<string | null>(null)
	const n8nWorkflow = ref<N8nWorkflowSnapshot | null>(null)
	const logs = ref<string[]>([])
	function addLog(message: string) {
		logs.value.unshift(`${new Date().toLocaleTimeString()} ${message}`)
		logs.value = logs.value.slice(0, 80)
	}
	const wsStatus = ref<'disconnected' | 'connecting' | 'connected'>('disconnected')
	const simulatorPresets = ref<Record<string, unknown> | null>(null)

	const demoFederatedEndpoint =
		import.meta.env.VITE_FEDERATED_COORDINATOR_DEMO_URL || 'http://localhost:8092/api/v1/coordinator'
	const demoSimulatorBaseUrl = import.meta.env.VITE_SENSOR_SIMULATOR_URL || 'http://localhost:8091'

	const showPipelineModal = ref(false)
	const showSourceModal = ref(false)
	const showConfigModal = ref(false)
	const huggingFaceIntegration = ref<HuggingFaceIntegrationStatus>({ configured: false })
	const hfModelCatalog = ref<HuggingFaceModelCatalogPage | null>(null)
	const hfModelSearch = ref('')
	const hfModelPage = ref(1)
	const hfModelLoading = ref(false)
	const hfManualModelId = ref('')
	const hfSelectedModelId = ref('')
	const hfSelectedModelDetails = ref<HuggingFaceModelSummary | null>(null)
	const hfTokenDraft = ref('')
	const showObservability = ref(false)
	const showSystemLog = ref(false)
	const showDataLakeModal = ref(false)
	const showExportTargetModal = ref(false)
	const showBackfillModal = ref(false)
	const editingExportTargetId = ref<string | null>(null)
	const lakeExportTab = ref<'overview' | 'browser' | 'export'>('overview')
	const showAdvancedExportJson = ref(false)

	const pipelineForm = reactive({ name: '', description: '' })
	const sourceForm = reactive({
		name: '',
		type: 'WEBSOCKET' as 'WEBSOCKET' | 'HTTP_POLLING',
		sensorKind: 'iot',
		mode: 'SIMULATED' as 'SIMULATED' | 'EXTERNAL',
		endpoint: '',
		pollIntervalMs: 5000,
		payloadPath: '',
		locationField: '',
		subscribeMessage: '',
		method: 'GET' as 'GET' | 'POST',
	})
	const exportForm = reactive({
		stage: 'business' as ExportStage,
		adapterType: 'json' as ExportAdapterKind,
		saveCredentials: false,
		limit: 100,
		targetName: '',
		isContinuous: false,
		cadenceSeconds: 60,
	})
	const backfillForm = reactive({
		stage: 'all' as 'raw' | 'cleaned' | 'business' | 'all',
		limit: 500,
	})
	const exportSettingsForm = reactive({
		jsonPrettyPrint: true,
		csvDelimiter: ',',
		postgresHost: '',
		postgresPort: 5432,
		postgresUsername: '',
		postgresPassword: '',
		postgresDatabase: '',
		postgresTable: 'smart_city_business',
		postgresIfExists: 'append' as 'append' | 'replace',
		mongodbUri: '',
		mongodbDatabase: '',
		mongodbCollection: 'smart_city_business',
		mongodbIfExists: 'append' as 'append' | 'replace',
		huggingFaceToken: '',
		huggingFaceRepoName: '',
		huggingFaceCommitMessage: '',
		huggingFacePrivate: false,
		advancedJson: '',
	})
	const streamConfig = reactive({
		coreUnitMode: 'manual' as 'manual' | 'auto',
		anomalyDetection: true,
		schemaValidation: true,
		autoCleaning: true,
		throughputLimit: 5,
		encryptionLevel: 'Standard',
	})
	const routingPreview = ref<Record<string, unknown> | null>(null)
	const runtimeForm = reactive({
		sourcePollIntervalMs: 3000,
		lakeWriteMode: 'append' as 'append' | 'object',
		exportCadenceSeconds: 60,
	})

	let refreshTimer: number | undefined

	const selectedPipeline = computed(() =>
		pipelines.value.find(pipeline => pipeline.id === selectedPipelineId.value),
	)

	const liveStream = createPipelineLiveStream(
		{
			events,
			latestProcessingResult,
			processingError,
			latestBackfill,
			wsStatus,
		},
		addLog,
	)

	const federated = usePipelineFederated({
		selectedPipelineId,
		selectedPipeline,
		pipelines,
		federatedRounds,
		saving,
		viewMode,
		demoFederatedEndpoint,
		addLog,
		loadDashboard,
	})

	const dataLake = usePipelineDataLake({
		selectedPipelineId,
		selectedPipeline,
		pipelines,
		dataLakes,
		dataLakeObjects,
		saving,
		showDataLakeModal,
		addLog,
	})

	const {
		federatedForm,
		federatedRoundForm,
		federatedUpdateForm,
		federatedAggregateForm,
		federatedConfig,
		activeFederatedRound,
		loadFederatedRounds,
		connectFederated,
		testFederatedConnection,
		disconnectFederatedConnection,
		startFederatedRoundFlow,
		submitFederatedRoundUpdate,
		aggregateFederatedRoundFlow,
		syncFederatedGlobalState,
		openFederatedWorkspace,
		syncFederatedForm,
	} = federated

	const {
		dataLakeForm,
		saveDataLake,
		testDataLake,
		disconnectDataLake,
		deleteDataLakeConnection,
		refreshLakeBrowser,
		replaceDataLake,
	} = dataLake
	const runningSourceCount = computed(
		() => sources.value.filter(source => source.status === 'RUNNING').length,
	)
	const activeDataLake = computed(() => {
		const selected = selectedPipeline.value?.dataLakeConnectionId
		return dataLakes.value.find(lake => lake.id === selected) ?? dataLakes.value.find(lake => lake.isDefault)
	})
	const throughput = computed(() => Number((runningSourceCount.value * 0.84 + 0.2).toFixed(2)))
	const sourceStatusText = computed(() =>
		sources.value.length ? `${runningSourceCount.value}/${sources.value.length} running` : 'No sources',
	)
	const recentEventsPreview = computed(() => events.value.slice(0, 5))
	const isApplyingAutoRouting = computed(() => autoRoutingPhase.value !== 'idle')
	const autoRoutingStatusMessage = computed(() => {
		if (autoRoutingPhase.value === 'matching') {
			return 'Matching sensor streams to the best trained models…'
		}
		if (autoRoutingPhase.value === 'saving') {
			return 'Saving adaptive auto mode…'
		}
		return null
	})
	const autoRoutingPolicy = computed((): AutoRoutingPolicySnapshot | null => {
		const policy = selectedPipeline.value?.streamConfig?.autoRoutingPolicy
		return policy && typeof policy === 'object' ? (policy as AutoRoutingPolicySnapshot) : null
	})
	const autoRoutingBindings = computed(() => autoRoutingPolicy.value?.bindings ?? [])
	const autoRoutingSummary = computed(() => autoRoutingPolicy.value?.summary ?? '')
	const lastAutoResolution = computed(() => {
		const resolution = selectedPipeline.value?.streamConfig?.lastAutoResolution
		return resolution && typeof resolution === 'object'
			? (resolution as { label?: string; modelId?: string; sensorKind?: string; reason?: string })
			: null
	})
	const activeModelLabel = computed(() => {
		if (autoRoutingPhase.value === 'matching') return 'Selecting best models…'
		const mode = streamConfig.coreUnitMode === 'auto' ? 'auto' : 'manual'
		if (mode === 'auto') {
			if (autoRoutingBindings.value.length > 0) {
				return autoRoutingBindings.value.map(binding => `${binding.sensorKind} → ${binding.label}`).join(' · ')
			}
			return 'Apply mode to configure routes'
		}
		const activeModelId = selectedPipeline.value?.activeModelId
		if (!activeModelId) return 'No model selected'
		if (activeModelId.startsWith('hf:')) return activeModelId.replace(/^hf:/, '')
		if (activeModelId.startsWith('research:')) return activeModelId.replace(/^research:/, 'Research · ')
		return activeModelId
	})
	const activeModelSubLabel = computed(() => {
		if (streamConfig.coreUnitMode !== 'auto') return null
		if (autoRoutingPhase.value === 'matching') return autoRoutingStatusMessage.value
		if (lastAutoResolution.value?.label) {
			return `Last live event: ${lastAutoResolution.value.sensorKind ?? 'stream'} → ${lastAutoResolution.value.label}`
		}
		if (autoRoutingSummary.value) return autoRoutingSummary.value
		return 'Each event is routed to a modality-safe checkpoint at runtime.'
	})
	const latestTrainingRun = computed(() => trainingRuns.value[0] ?? null)
	const processingModelCount = computed(() => hfModelCatalog.value?.total ?? 0)
	const usesSavedHuggingFaceToken = computed(() => huggingFaceIntegration.value.configured)
	const linkedDataLake = computed(() =>
		dataLakes.value.find(lake => lake.id === selectedPipeline.value?.dataLakeConnectionId) ?? null,
	)
	const selectedExportAdapter = computed(() =>
		exportAdapters.value.find(adapter => adapter.type === exportForm.adapterType) ?? null,
	)
	const exportCredentialHint = computed(() => {
		switch (exportForm.adapterType) {
			case 'huggingface':
				return usesSavedHuggingFaceToken.value
					? `Using saved organization token${huggingFaceIntegration.value.username ? ` (@${huggingFaceIntegration.value.username})` : ''}.`
					: 'Save a Hugging Face token in Settings to reuse it across exports and the core unit.'
			case 'postgres':
				return 'Host, port, username, password, and database can be saved securely.'
			case 'mongodb':
				return 'URI and database can be saved securely.'
			default:
				return 'This adapter does not require reusable credentials.'
		}
	})
	const exportSettingsPreview = computed(() =>
		buildExportSettingsPreview(exportForm.adapterType, exportSettingsForm),
	)
	const supportsSavedCredentials = computed(() =>
		['huggingface', 'postgres', 'mongodb'].includes(exportForm.adapterType),
	)
	const sourceMixSummary = computed(() => {
		const external = sources.value.filter(source => source.mode === 'EXTERNAL').length
		const simulated = sources.value.length - external
		return `${external} external / ${simulated} simulated`
	})
	const pipelineRuntimeStatus = computed(() => selectedPipeline.value?.status ?? 'DRAFT')
	const pipelineRuntimeLabel = computed(() => {
		switch (pipelineRuntimeStatus.value) {
			case 'ACTIVE':
				return 'Live stream'
			case 'PAUSED':
				return 'Paused'
			case 'ERROR':
				return 'Error'
			default:
				return 'Stopped'
		}
	})
	const isPipelineLive = computed(() => pipelineRuntimeStatus.value === 'ACTIVE')
	const continuousExportTargets = computed(() =>
		exportTargets.value.filter(target => target.isContinuous && target.status !== 'ARCHIVED'),
	)
	const exportHealthSummary = computed(() => {
		const errorTargets = exportTargets.value.filter(target => target.lastError && target.status !== 'ARCHIVED')
		const lastRun = exportRuns.value[0]
		return {
			continuous: continuousExportTargets.value.length,
			errors: errorTargets.length,
			lastStatus: lastRun?.status ?? null,
			lastAdapter: lastRun?.adapterType ?? null,
			lastAt: lastRun?.finishedAt ?? lastRun?.startedAt ?? null,
		}
	})
	const pipelineFlowHint = computed(() =>
		buildPipelineFlowHint({
			isLive: isPipelineLive.value,
			runningSourceCount: runningSourceCount.value,
			recentEventCount: events.value.length,
			continuousExportTargets: continuousExportTargets.value,
			exportErrorCount: exportHealthSummary.value.errors,
		}),
	)
	const simulatorEndpointHint = computed(() => {
		if (sourceForm.mode !== 'SIMULATED') return ''
		const presets = simulatorPresets.value?.presets as Record<string, { httpPollUrl?: string; websocketUrl?: string }> | undefined
		const preset = presets?.[sourceForm.sensorKind]
		if (sourceForm.type === 'WEBSOCKET') {
			return preset?.websocketUrl ?? `${demoSimulatorBaseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=${sourceForm.sensorKind}`
		}
		return preset?.httpPollUrl ?? `${demoSimulatorBaseUrl}/api/v1/poll?sensorKind=${sourceForm.sensorKind}&limit=1`
	})

	watch(selectedPipelineId, id => {
		if (id) void loadDashboard(id)
	})

	watch(
		() => exportForm.adapterType,
		adapterType => {
			refreshAdapterDefaults(adapterType)
		},
		{ immediate: true },
	)

	watch(
		() => exportForm.stage,
		stage => {
			refreshStageDependentExportFields(stage)
		},
	)

	watch(exportPreviewStage, () => {
		void loadExportPreview()
	})

	onMounted(async () => {
		federated.bootstrapFederatedEndpoint()
		try {
			const presetResponse = await apiClient.getSimulatorPresets()
			simulatorPresets.value = presetResponse.data ?? null
		} catch {
			simulatorPresets.value = null
		}
		await loadPipelines()
		await loadHuggingFaceIntegration()
		refreshTimer = window.setInterval(refreshEvents, 5000)
	})

	onUnmounted(() => {
		if (refreshTimer) window.clearInterval(refreshTimer)
		liveStream.disconnect()
	})

	async function loadHuggingFaceIntegration() {
		try {
			const response = await apiClient.getHuggingFaceStatus()
			huggingFaceIntegration.value = response.data ?? { configured: false }
			if (huggingFaceIntegration.value.configured && exportForm.adapterType === 'huggingface') {
				exportForm.saveCredentials = true
			}
		} catch {
			huggingFaceIntegration.value = { configured: false }
		}
	}

	async function saveHuggingFaceToken(token: string) {
		if (!token.trim()) return toast.error('Enter a Hugging Face token')
		const response = await apiClient.saveHuggingFaceToken(token.trim())
		huggingFaceIntegration.value = response.data ?? { configured: true }
		hfTokenDraft.value = ''
		exportForm.saveCredentials = true
		await loadHuggingFaceCatalog(1)
		toast.success('Hugging Face token saved for this organization')
	}

	async function removeHuggingFaceToken() {
		await apiClient.removeHuggingFaceToken()
		huggingFaceIntegration.value = { configured: false }
		toast.success('Hugging Face token removed')
	}

	async function loadHuggingFaceCatalog(page = hfModelPage.value) {
		if (!huggingFaceIntegration.value.configured) {
			hfModelCatalog.value = null
			return
		}
		hfModelLoading.value = true
		try {
			const response = await apiClient.listHuggingFaceModels({
				search: hfModelSearch.value.trim() || undefined,
				page,
				limit: 8,
			})
			hfModelCatalog.value = response.data ?? null
			hfModelPage.value = page
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load Hugging Face models')
		} finally {
			hfModelLoading.value = false
		}
	}

	async function selectHuggingFaceModel(model: HuggingFaceModelSummary) {
		hfSelectedModelId.value = model.modelId
		hfManualModelId.value = model.modelId
		hfSelectedModelDetails.value = model
	}

	async function resolveManualHuggingFaceModel() {
		const modelId = hfManualModelId.value.trim()
		if (!modelId.includes('/')) {
			toast.error('Use the format username/model-name')
			return
		}
		try {
			const response = await apiClient.getHuggingFaceModel(modelId)
			if (response.data) {
				hfSelectedModelId.value = response.data.modelId
				hfSelectedModelDetails.value = response.data
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not resolve model')
		}
	}

	async function loadPipelines() {
		loading.value = true
		try {
			const response = await apiClient.listSmartCityPipelines()
			const adapterResponse = await apiClient.getExportAdapters()
			exportAdapters.value = adapterResponse.data ?? []
			if (!exportForm.targetName) exportForm.targetName = 'Business export'
			pipelines.value = response.data ?? []
			const queryPipelineId =
				typeof route.query.pipelineId === 'string' ? route.query.pipelineId : ''
			const preferredId =
				queryPipelineId && pipelines.value.some(pipeline => pipeline.id === queryPipelineId)
					? queryPipelineId
					: pipelines.value[0]?.id ?? ''
			if (!selectedPipelineId.value || queryPipelineId) selectedPipelineId.value = preferredId
			if (!selectedPipelineId.value) addLog('[READY] Create a pipeline to connect sources')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load pipelines')
		} finally {
			loading.value = false
		}
	}

	async function loadDashboard(pipelineId: string) {
		loading.value = true
		try {
			const response = await apiClient.getSmartCityDashboard(pipelineId)
			if (!response.data) return
			const dashboard = response.data
			replacePipeline(dashboard.pipeline)
			sources.value = dashboard.sources
			events.value = dashboard.events
			dataLakes.value = dashboard.dataLakes
			const [targetsResponse, runsResponse] = await Promise.all([
				apiClient.listSmartCityExportTargets(pipelineId),
				apiClient.listSmartCityExportRuns(pipelineId),
			])
			exportTargets.value = targetsResponse.data ?? []
			exportRuns.value = runsResponse.data ?? []
			await loadExportPreview(pipelineId)
			const lakeObjectsResponse = await apiClient.listSmartCityDataLakeObjects(pipelineId)
			dataLakeObjects.value = lakeObjectsResponse.data?.objects ?? []
			Object.assign(streamConfig, dashboard.pipeline.streamConfig)
			syncRuntimeForm(dashboard.pipeline.streamConfig)
			syncFederatedForm(dashboard.pipeline)
			await loadFederatedRounds(pipelineId)
			await loadObservability(pipelineId)
			await loadN8nWorkflow()
			await loadModels()
			liveStream.connect(pipelineId)
			addLog(`[SYNC] ${dashboard.sources.length} sources, ${dashboard.events.length} recent events`)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load dashboard')
		} finally {
			loading.value = false
		}
	}

	async function loadExportPreview(pipelineId = selectedPipelineId.value) {
		if (!pipelineId) {
			exportPreview.value = null
			return
		}
		try {
			const response = await apiClient.getSmartCityExportPreview(
				pipelineId,
				exportPreviewStage.value,
				8,
			)
			exportPreview.value = response.data ?? null
		} catch {
			exportPreview.value = null
		}
	}

	async function refreshEvents() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.listSensorEvents(selectedPipelineId.value, 25)
			events.value = response.data ?? []
		} catch {
			// Silent refresh keeps the page stable during auth/org switches.
		}
	}

	async function loadN8nWorkflow() {
		if (!selectedPipelineId.value) return
		const response = await apiClient.getSmartCityN8nWorkflow(selectedPipelineId.value)
		n8nWorkflow.value = response.data
	}

	async function loadModels() {
		const [artifactResponse, runResponse] = await Promise.all([
			apiClient.listModelArtifacts(),
			apiClient.listModelTrainingRuns(),
		])
		models.value = artifactResponse.data ?? []
		trainingRuns.value = runResponse.data ?? []
		const researchResponse = await apiClient.listResearchModels()
		researchModels.value = researchResponse.data ?? []
	}

	async function loadObservability(pipelineId: string) {
		const response = await apiClient.getSmartCityObservability(pipelineId)
		observability.value = response.data ?? null
	}

	async function createPipeline() {
		if (!pipelineForm.name.trim()) return toast.error('Pipeline name is required')
		saving.value = true
		try {
			const response = await apiClient.createSmartCityPipeline({
				name: pipelineForm.name,
				description: pipelineForm.description || undefined,
			})
			if (response.data) {
				pipelines.value.unshift(response.data)
				selectedPipelineId.value = response.data.id
				pipelineForm.name = ''
				pipelineForm.description = ''
				showPipelineModal.value = false
				addLog(`[PIPELINE] Created ${response.data.name}`)
				toast.success('Pipeline created')
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not create pipeline')
		} finally {
			saving.value = false
		}
	}

	async function archivePipeline() {
		if (!selectedPipeline.value || !window.confirm(`Archive "${selectedPipeline.value.name}"?`)) return
		try {
			await apiClient.deleteSmartCityPipeline(selectedPipeline.value.id)
			pipelines.value = pipelines.value.filter(pipeline => pipeline.id !== selectedPipeline.value?.id)
			selectedPipelineId.value = pipelines.value[0]?.id ?? ''
			sources.value = []
			events.value = []
			addLog('[PIPELINE] Archived pipeline')
			toast.success('Pipeline archived')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not archive pipeline')
		}
	}

	async function addSource() {
		if (!selectedPipelineId.value) return toast.error('Create a pipeline first')
		if (!sourceForm.name.trim()) return toast.error('Source name is required')
		if (sourceForm.mode === 'EXTERNAL' && !sourceForm.endpoint.trim()) {
			return toast.error('External sources require an endpoint')
		}
		saving.value = true
		try {
			const response = await apiClient.createSensorSource(selectedPipelineId.value, {
				name: sourceForm.name,
				type: sourceForm.type,
				sensorKind: sourceForm.sensorKind,
				mode: sourceForm.mode,
				endpoint: sourceForm.endpoint || undefined,
				pollIntervalMs: sourceForm.pollIntervalMs,
				connectionConfig:
					sourceForm.mode === 'EXTERNAL'
						? {
								payloadPath: sourceForm.payloadPath || undefined,
								locationField: sourceForm.locationField || undefined,
								subscribeMessage:
									sourceForm.type === 'WEBSOCKET' ? sourceForm.subscribeMessage || undefined : undefined,
								method: sourceForm.type === 'HTTP_POLLING' ? sourceForm.method : undefined,
							}
						: undefined,
			})
			if (response.data) {
				sources.value.push(response.data)
				resetSourceForm()
				showSourceModal.value = false
				addLog(`[SOURCE] Added ${response.data.name}`)
				await loadN8nWorkflow()
				toast.success('Source added')
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not add source')
		} finally {
			saving.value = false
		}
	}

	async function startSource(source: SensorSource) {
		try {
			const response = await apiClient.startSensorSource(source.id)
			replaceSource(response.data)
			await refreshEvents()
			addLog(`[STREAM] Started ${source.name}`)
			toast.success(`${source.name} started`)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not start source')
		}
	}

	async function stopSource(source: SensorSource) {
		try {
			const response = await apiClient.stopSensorSource(source.id)
			replaceSource(response.data)
			addLog(`[STREAM] Stopped ${source.name}`)
			toast.success(`${source.name} stopped`)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not stop source')
		}
	}

	async function testSource(source: SensorSource) {
		try {
			const response = await apiClient.testSensorSource(source.id)
			handleSourceTestResult(source, response.data)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not test source')
		}
	}

	async function removeSource(source: SensorSource) {
		if (!window.confirm(`Delete "${source.name}"?`)) return
		try {
			await apiClient.deleteSensorSource(source.id)
			sources.value = sources.value.filter(item => item.id !== source.id)
			await loadN8nWorkflow()
			addLog(`[SOURCE] Removed ${source.name}`)
			toast.success('Source deleted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not delete source')
		}
	}

	function openConfigModal() {
		const active = selectedPipeline.value?.activeModelId ?? ''
		streamConfig.coreUnitMode =
			selectedPipeline.value?.streamConfig?.coreUnitMode === 'auto' ? 'auto' : 'manual'
		hfSelectedModelId.value = active.startsWith('hf:') ? active.replace(/^hf:/, '') : ''
		hfManualModelId.value = hfSelectedModelId.value
		hfSelectedModelDetails.value = null
		hfModelSearch.value = ''
		hfModelPage.value = 1
		void loadHuggingFaceIntegration().then(() => loadHuggingFaceCatalog(1))
		void loadRoutingPreview()
		showConfigModal.value = true
	}

	async function loadRoutingPreview() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.previewCoreUnitRouting(selectedPipelineId.value)
			routingPreview.value = (response.data as Record<string, unknown> | undefined) ?? null
		} catch {
			routingPreview.value = null
		}
	}

	async function saveConfiguration() {
		if (!selectedPipelineId.value) return
		const previousModelId = selectedPipeline.value?.activeModelId ?? ''
		const nextModelId = hfSelectedModelId.value.trim() || hfManualModelId.value.trim()
		saving.value = true
		autoRoutingPhase.value = streamConfig.coreUnitMode === 'auto' ? 'saving' : 'idle'
		try {
			const response = await apiClient.updateSmartCityPipeline(selectedPipelineId.value, {
				streamConfig: {
					...streamConfig,
					coreUnitMode: streamConfig.coreUnitMode,
				},
			})
			replacePipeline(response.data)

			if (streamConfig.coreUnitMode === 'auto') {
				autoRoutingPhase.value = 'matching'
				const policyResponse = await apiClient.buildCoreUnitAutoPolicy(selectedPipelineId.value)
				replacePipeline(policyResponse.data)
				Object.assign(streamConfig, policyResponse.data?.streamConfig ?? {})
				await loadRoutingPreview()
				addLog('[MODEL] Enabled adaptive auto routing on core unit')
			} else if (nextModelId) {
				const normalizedNext = `hf:${nextModelId}`
				if (normalizedNext !== previousModelId) {
					const deployResponse = await apiClient.deployHuggingFaceModel(
						selectedPipelineId.value,
						nextModelId,
					)
					replacePipeline(deployResponse.data)
					addLog(`[MODEL] Selected Hugging Face model ${nextModelId}`)
				}
			}

			showConfigModal.value = false
			addLog(`[CONFIG] Saved processing settings (${streamConfig.throughputLimit} GB/s limit)`)
			toast.success(
				streamConfig.coreUnitMode === 'auto'
					? 'Adaptive auto routing configured'
					: 'Processing configuration saved',
			)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not save configuration')
		} finally {
			autoRoutingPhase.value = 'idle'
			saving.value = false
		}
	}

	function syncRuntimeForm(config?: Record<string, unknown>) {
		const runtime = (config?.runtime ?? {}) as Record<string, unknown>
		runtimeForm.sourcePollIntervalMs =
			typeof runtime.sourcePollIntervalMs === 'number' ? runtime.sourcePollIntervalMs : 3000
		runtimeForm.lakeWriteMode = runtime.lakeWriteMode === 'object' ? 'object' : 'append'
		runtimeForm.exportCadenceSeconds =
			typeof runtime.exportCadenceSeconds === 'number' ? runtime.exportCadenceSeconds : 60
	}

	async function startPipelineRuntime() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.startSmartCityPipeline(selectedPipelineId.value, {
				sourcePollIntervalMs: runtimeForm.sourcePollIntervalMs,
			})
			replacePipeline(response.data)
			await loadDashboard(selectedPipelineId.value)
			addLog('[RUNTIME] Pipeline started')
			toast.success('Pipeline started')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not start pipeline')
		} finally {
			saving.value = false
		}
	}

	async function stopPipelineRuntime() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.stopSmartCityPipeline(selectedPipelineId.value)
			replacePipeline(response.data)
			await loadDashboard(selectedPipelineId.value)
			addLog('[RUNTIME] Pipeline stopped')
			toast.success('Pipeline stopped')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not stop pipeline')
		} finally {
			saving.value = false
		}
	}

	async function resumePipelineRuntime() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.resumeSmartCityPipeline(selectedPipelineId.value)
			replacePipeline(response.data)
			await loadDashboard(selectedPipelineId.value)
			addLog('[RUNTIME] Pipeline resumed')
			toast.success('Pipeline resumed')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not resume pipeline')
		} finally {
			saving.value = false
		}
	}

	async function savePipelineRuntime() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.updateSmartCityPipelineRuntime(selectedPipelineId.value, {
				sourcePollIntervalMs: runtimeForm.sourcePollIntervalMs,
				lakeWriteMode: runtimeForm.lakeWriteMode,
				exportCadenceSeconds: runtimeForm.exportCadenceSeconds,
			})
			replacePipeline(response.data)
			syncRuntimeForm(response.data?.streamConfig as Record<string, unknown> | undefined)
			addLog('[RUNTIME] Updated interval and lake write mode')
			toast.success('Runtime settings saved')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not save runtime settings')
		} finally {
			saving.value = false
		}
	}

	async function pollHttpFeed() {
		if (!selectedPipelineId.value) return
		const response = await apiClient.pollSmartCityHttpStream(selectedPipelineId.value)
		if (response.data) {
			events.value = response.data.latest
			addLog(`[HTTP] Poll generated ${response.data.generated.length} events`)
		}
	}

	async function trainAstanaModel() {
		if (!selectedPipelineId.value) return toast.error('Create or select a pipeline first')
		saving.value = true
		try {
			const response = await apiClient.trainSmartCityModel({
				name: `${selectedPipeline.value?.name ?? 'Astana'} traffic baseline`,
				pipelineId: selectedPipelineId.value,
				modelType: 'TRAFFIC_BASELINE',
			})
			if (response.data) {
				trainingRuns.value.unshift(response.data)
				models.value = [...(response.data.artifacts ?? []), ...models.value]
				addLog(`[MODEL] Training ${response.data.status.toLowerCase()} for ${response.data.name}`)
				toast.success('Model trained from Astana CSV')
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not train model')
		} finally {
			saving.value = false
		}
	}

	async function promoteModel(model: ModelArtifact) {
		const response = await apiClient.promoteSmartCityModel(model.id)
		replaceModel(response.data)
		addLog(`[MODEL] Promoted ${model.name}`)
	}

	async function deployModel(model: ModelArtifact) {
		if (!selectedPipelineId.value) return
		const response = await apiClient.deploySmartCityModel(selectedPipelineId.value, model.id)
		replacePipeline(response.data)
		addLog(`[MODEL] Selected ${model.name} for processing`)
		toast.success('Model selected for pipeline')
	}

	async function deployResearchModel(model: ResearchModel) {
		if (!selectedPipelineId.value) return
		const response = await apiClient.deployResearchModel(selectedPipelineId.value, model.run)
		replacePipeline(response.data)
		addLog(`[MODEL] Selected research checkpoint ${model.run}`)
		toast.success('Research model selected')
	}

	async function testProcessingUnit() {
		if (!selectedPipelineId.value) return
		const response = await apiClient.testProcessing(selectedPipelineId.value, {
			averageSpeedKph: 52,
			vehicleCount: 96,
			Latitude: 51.12,
			Longitude: 71.45,
			Traffic_Density: 76,
		})
		latestProcessingResult.value = response.data
		if (response.data?.error) {
			processingError.value = response.data.error
		} else {
			processingError.value = null
		}
		addLog('[PROCESS] Manual processing test completed')
	}

	async function exportStageToAdapter() {
		if (!selectedPipelineId.value) return
		const validationError = validateExportTargetFormState()
		if (validationError) return toast.error(validationError)
		saving.value = true
		try {
			const response = await apiClient.exportSmartCityPipelineStage(selectedPipelineId.value, {
				stage: exportForm.stage,
				adapterType: exportForm.adapterType,
				settings: buildExportSettingsPayload(),
				saveCredentials: exportForm.saveCredentials,
				limit: exportForm.limit,
			})
			addLog(
				`[EXPORT] ${response.data?.stage ?? exportForm.stage} -> ${response.data?.destination ?? exportForm.adapterType}`,
			)
			const runsResponse = await apiClient.listSmartCityExportRuns(selectedPipelineId.value)
			exportRuns.value = runsResponse.data ?? []
			const lakeObjectsResponse = await apiClient.listSmartCityDataLakeObjects(selectedPipelineId.value)
			dataLakeObjects.value = lakeObjectsResponse.data?.objects ?? []
			toast.success(response.data?.message ?? 'Stage export completed')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not export pipeline stage')
		} finally {
			saving.value = false
		}
	}

	async function saveExportTarget() {
		if (!selectedPipelineId.value) return
		if (!exportForm.targetName.trim()) return toast.error('Export target name is required')
		const validationError = validateExportTargetFormState()
		if (validationError) return toast.error(validationError)

		saving.value = true
		try {
			const payload = {
				name: exportForm.targetName.trim(),
				stage: exportForm.stage,
				adapterType: exportForm.adapterType,
				settings: buildExportSettingsPayload(),
				saveCredentials: exportForm.saveCredentials,
				isContinuous: exportForm.isContinuous,
				cadenceSeconds: exportForm.cadenceSeconds,
			}
			if (editingExportTargetId.value) {
				const response = await apiClient.updateSmartCityExportTarget(editingExportTargetId.value, payload)
				replaceExportTarget(response.data)
				addLog(`[EXPORT] Updated target ${response.data?.name ?? exportForm.targetName}`)
				toast.success('Export target updated')
			} else {
				const response = await apiClient.createSmartCityExportTarget(selectedPipelineId.value, payload)
				if (response.data) {
					exportTargets.value = [response.data, ...exportTargets.value.filter(item => item.id !== response.data?.id)]
					addLog(`[EXPORT] Saved target ${response.data.name}`)
					toast.success('Export target saved')
				}
			}
			editingExportTargetId.value = null
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not save export target')
		} finally {
			saving.value = false
		}
	}

	async function runExportTarget(target: SmartCityExportTarget) {
		try {
			await apiClient.runSmartCityExportTarget(target.id)
			addLog(`[EXPORT] Queued target ${target.name}`)
			await new Promise(resolve => window.setTimeout(resolve, 2500))
			if (selectedPipelineId.value) {
				const runsResponse = await apiClient.listSmartCityExportRuns(selectedPipelineId.value)
				exportRuns.value = runsResponse.data ?? []
				await loadExportPreview(selectedPipelineId.value)
			}
			toast.success('Export target run finished')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not queue export target')
		}
	}

	async function deleteExportTarget(target: SmartCityExportTarget) {
		if (!window.confirm(`Delete export target "${target.name}"?`)) return
		try {
			await apiClient.deleteSmartCityExportTarget(target.id)
			exportTargets.value = exportTargets.value.filter(item => item.id !== target.id)
			if (editingExportTargetId.value === target.id) {
				editingExportTargetId.value = null
				showExportTargetModal.value = false
			}
			addLog(`[EXPORT] Deleted target ${target.name}`)
			toast.success('Export target deleted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not delete export target')
		}
	}

	async function deleteEditingExportTarget() {
		const target = exportTargets.value.find(item => item.id === editingExportTargetId.value)
		if (!target) return toast.error('Export target not found')
		await deleteExportTarget(target)
	}

	async function backfillDataLakeStages() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.backfillSmartCityPipeline(selectedPipelineId.value, {
				stage: backfillForm.stage,
				limit: backfillForm.limit,
			})
			latestBackfill.value = response.data ?? null
			await refreshLakeBrowser()
			addLog(
				`[BACKFILL] ${response.data?.scannedEvents ?? 0} events replayed for ${response.data?.stage ?? backfillForm.stage}`,
			)
			toast.success(response.data?.message ?? 'Backfill completed')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not run backfill')
		} finally {
			saving.value = false
		}
	}

	async function toggleContinuousTarget(target: SmartCityExportTarget) {
		try {
			const response = await apiClient.updateSmartCityExportTarget(target.id, {
				isContinuous: !target.isContinuous,
				cadenceSeconds: target.cadenceSeconds,
			})
			replaceExportTarget(response.data)
			addLog(
				`[EXPORT] ${response.data?.isContinuous ? 'Enabled' : 'Disabled'} continuous target ${target.name}`,
			)
			toast.success('Export target updated')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not update export target')
		}
	}

	function replacePipeline(pipeline?: SmartCityPipeline) {
		replaceInList(pipelines.value, pipeline)
	}

	function replaceSource(source?: SensorSource) {
		replaceInList(sources.value, source)
	}

	function replaceModel(model?: ModelArtifact) {
		replaceInList(models.value, model)
	}

	function replaceExportTarget(target?: SmartCityExportTarget) {
		replaceInList(exportTargets.value, target)
	}

	function refreshAdapterDefaults(adapterType: ExportAdapterKind = exportForm.adapterType) {
		Object.assign(
			exportSettingsForm,
			applyAdapterDefaults({ ...exportForm, adapterType }, exportSettingsForm),
		)
	}

	function refreshStageDependentExportFields(stage: ExportStage = exportForm.stage) {
		Object.assign(exportSettingsForm, syncStageDependentExportFields(stage, exportSettingsForm))
	}

	function buildExportSettingsPayload(): Record<string, unknown> {
		return buildExportSettings(exportForm, exportSettingsForm, () => toast.error('Advanced export JSON is invalid'))
	}

	function validateExportTargetFormState() {
		return validateExportTargetForm({
			form: exportForm,
			settings: exportSettingsForm,
			usesSavedHuggingFaceToken: usesSavedHuggingFaceToken.value,
			isEditing: Boolean(editingExportTargetId.value),
		})
	}

	function resetExportTargetForm() {
		editingExportTargetId.value = null
		exportForm.targetName = ''
		exportForm.stage = 'business'
		exportForm.adapterType = 'json'
		exportForm.saveCredentials = usesSavedHuggingFaceToken.value
		exportForm.limit = 100
		exportForm.isContinuous = false
		exportForm.cadenceSeconds = 60
		exportSettingsForm.advancedJson = ''
		refreshAdapterDefaults('json')
		refreshStageDependentExportFields('business')
	}

	function openExportTargetModal(target?: SmartCityExportTarget) {
		if (target) {
			loadExportTargetIntoForm(target, { silent: true })
		} else {
			resetExportTargetForm()
			if (!exportForm.targetName) exportForm.targetName = 'Business export'
		}
		showExportTargetModal.value = true
	}

	function loadExportTargetIntoForm(
		target: SmartCityExportTarget,
		options: { silent?: boolean } = {},
	) {
		editingExportTargetId.value = target.id
		exportForm.targetName = target.name
		exportForm.stage = target.stage
		exportForm.adapterType = target.adapterType as ExportAdapterKind
		exportForm.saveCredentials = target.saveCredentials
		exportForm.isContinuous = target.isContinuous
		exportForm.cadenceSeconds = target.cadenceSeconds
		Object.assign(
			exportSettingsForm,
			applyExportSettingsToForm(
				target.adapterType as ExportAdapterKind,
				target.stage,
				target.settingsJson ?? {},
				exportSettingsForm,
			),
		)
		if (!options.silent) {
			addLog(`[EXPORT] Loaded target ${target.name} into form`)
			toast.success('Export target loaded into form')
		}
	}

	function resetSourceForm() {
		Object.assign(sourceForm, {
			name: '',
			type: 'WEBSOCKET',
			sensorKind: 'iot',
			mode: 'SIMULATED',
			endpoint: '',
			pollIntervalMs: 5000,
			payloadPath: '',
			locationField: '',
			subscribeMessage: '',
			method: 'GET',
		})
	}

	function handleSourceTestResult(source: SensorSource, result?: SensorSourceTestResult) {
		if (!result) {
			toast.success('Source test completed')
			return
		}
		const event = result.event
		if (event && typeof event === 'object' && 'id' in event) {
			events.value.unshift(event as SensorEvent)
		}
		addLog(`[TEST] ${source.name}: ${result.summary ?? 'test completed'}`)
		toast.success(result.summary ?? 'Source test completed')
	}

	return {
		viewMode,
		activeStage,
		loading,
		saving,
		pipelines,
		selectedPipelineId,
		sources,
		events,
		dataLakes,
		exportAdapters,
		exportTargets,
		exportRuns,
		exportPreview,
		exportPreviewStage,
		dataLakeObjects,
		latestBackfill,
		federatedRounds,
		observability,
		models,
		trainingRuns,
		researchModels,
		latestProcessingResult,
		processingError,
		n8nWorkflow,
		logs,
		wsStatus,
		simulatorPresets,
		demoFederatedEndpoint,
		demoSimulatorBaseUrl,
		showPipelineModal,
		showSourceModal,
		showConfigModal,
		huggingFaceIntegration,
		hfModelCatalog,
		hfModelSearch,
		hfModelPage,
		hfModelLoading,
		hfManualModelId,
		hfSelectedModelId,
		hfSelectedModelDetails,
		hfTokenDraft,
		usesSavedHuggingFaceToken,
		showObservability,
		showSystemLog,
		showDataLakeModal,
		showExportTargetModal,
		showBackfillModal,
		editingExportTargetId,
		lakeExportTab,
		showAdvancedExportJson,
		pipelineForm,
		sourceForm,
		dataLakeForm,
		exportForm,
		backfillForm,
		exportSettingsForm,
		streamConfig,
		routingPreview,
		runtimeForm,
		federatedForm,
		federatedRoundForm,
		federatedUpdateForm,
		federatedAggregateForm,
		selectedPipeline,
		runningSourceCount,
		activeDataLake,
		throughput,
		sourceStatusText,
		recentEventsPreview,
		activeModelLabel,
		activeModelSubLabel,
		isApplyingAutoRouting,
		autoRoutingPhase,
		autoRoutingStatusMessage,
		autoRoutingBindings,
		autoRoutingSummary,
		lastAutoResolution,
		latestTrainingRun,
		processingModelCount,
		linkedDataLake,
		selectedExportAdapter,
		exportCredentialHint,
		exportSettingsPreview,
		supportsSavedCredentials,
		federatedConfig,
		activeFederatedRound,
		sourceMixSummary,
		pipelineRuntimeStatus,
		pipelineRuntimeLabel,
		isPipelineLive,
		pipelineFlowHint,
		exportHealthSummary,
		continuousExportTargets,
		simulatorEndpointHint,
		loadPipelines,
		loadDashboard,
		refreshEvents,
		loadN8nWorkflow,
		loadModels,
		loadFederatedRounds,
		loadObservability,
		createPipeline,
		archivePipeline,
		addSource,
		startSource,
		stopSource,
		testSource,
		removeSource,
		openConfigModal,
		saveConfiguration,
		startPipelineRuntime,
		stopPipelineRuntime,
		resumePipelineRuntime,
		savePipelineRuntime,
		loadHuggingFaceIntegration,
		loadHuggingFaceCatalog,
		saveHuggingFaceToken,
		removeHuggingFaceToken,
		selectHuggingFaceModel,
		resolveManualHuggingFaceModel,
		saveDataLake,
		testDataLake,
		disconnectDataLake,
		deleteDataLakeConnection,
		pollHttpFeed,
		trainAstanaModel,
		promoteModel,
		deployModel,
		deployResearchModel,
		testProcessingUnit,
		exportStageToAdapter,
		saveExportTarget,
		openExportTargetModal,
		resetExportTargetForm,
		runExportTarget,
		loadExportPreview,
		deleteExportTarget,
		deleteEditingExportTarget,
		refreshLakeBrowser,
		backfillDataLakeStages,
		formatBytes,
		toggleContinuousTarget,
		connectFederated,
		testFederatedConnection,
		disconnectFederatedConnection,
		startFederatedRoundFlow,
		submitFederatedRoundUpdate,
		aggregateFederatedRoundFlow,
		syncFederatedGlobalState,
		openFederatedWorkspace,
		loadExportTargetIntoForm,
		addLog,
		resetSourceForm,
		handleSourceTestResult,
		sourceIcon,
		formatDateTime,
	}
}
