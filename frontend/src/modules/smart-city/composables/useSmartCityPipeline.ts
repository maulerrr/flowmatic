import {
	Activity,
	Brain,
	Cpu,
	Database,
	HardDrive,
	LayoutDashboard,
	Network,
	Plus,
	Radio,
	RefreshCw,
	Router,
	Save,
	Trash2,
	Video,
	Wifi,
	Workflow,
	X,
	Zap,
} from 'lucide-vue-next'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
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
	type SmartCityExportTarget,
	type FederatedConnectionState,
	type ModelArtifact,
	type ModelTrainingRun,
	type ResearchModel,
	type SensorEvent,
	type SensorSource,
	type SensorSourceTestResult,
	type SmartCityPipeline,
} from '@/api/client'

export type PipelineStage = 'sources' | 'processing' | 'lake' | 'federated'
export type ExportStage = 'raw' | 'cleaned' | 'business'
export type ExportAdapterKind = 'json' | 'csv' | 'postgres' | 'mongodb' | 'huggingface'

export function useSmartCityPipeline() {
	const viewMode = ref<'dashboard' | 'workflow'>('dashboard')
	const activeStage = ref<'sources' | 'processing' | 'lake' | 'federated'>('sources')
	const loading = ref(true)
	const saving = ref(false)
	const pipelines = ref<SmartCityPipeline[]>([])
	const selectedPipelineId = ref('')
	const sources = ref<SensorSource[]>([])
	const events = ref<SensorEvent[]>([])
	const dataLakes = ref<DataLakeConnection[]>([])
	const exportAdapters = ref<ExportAdapterMetadata[]>([])
	const exportTargets = ref<SmartCityExportTarget[]>([])
	const exportRuns = ref<SmartCityExportRun[]>([])
	const dataLakeObjects = ref<DataLakeObjectGroup[]>([])
	const latestBackfill = ref<SmartCityBackfillResult | null>(null)
	const federatedRounds = ref<FederatedRound[]>([])
	const observability = ref<SmartCityObservability | null>(null)
	const models = ref<ModelArtifact[]>([])
	const trainingRuns = ref<ModelTrainingRun[]>([])
	const researchModels = ref<ResearchModel[]>([])
	const latestProcessingResult = ref<any>(null)
	const n8nWorkflow = ref<any>(null)
	const logs = ref<string[]>([])
	const wsStatus = ref<'disconnected' | 'connecting' | 'connected'>('disconnected')
	const simulatorPresets = ref<Record<string, unknown> | null>(null)

	const demoFederatedEndpoint =
		import.meta.env.VITE_FEDERATED_COORDINATOR_DEMO_URL || 'http://localhost:8092/api/v1/coordinator'
	const demoSimulatorBaseUrl = import.meta.env.VITE_SENSOR_SIMULATOR_URL || 'http://localhost:8091'

	const showPipelineModal = ref(false)
	const showSourceModal = ref(false)
	const showConfigModal = ref(false)
	const showDataLakeModal = ref(false)
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
	const dataLakeForm = reactive({
		name: '',
		provider: 'CUSTOM_S3' as DataLakeConnection['provider'],
		bucket: '',
		region: '',
		endpoint: '',
		basePrefix: '',
		accessKey: '',
		secretKey: '',
		isDefault: true,
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
		huggingFaceFileName: 'smart_city_business.csv',
		huggingFaceCommitMessage: '',
		huggingFacePrivate: false,
		advancedJson: '',
	})
	const streamConfig = reactive({
		anomalyDetection: true,
		schemaValidation: true,
		autoCleaning: true,
		throughputLimit: 5,
		encryptionLevel: 'Standard',
	})
	const federatedForm = reactive({
		protocol: 'HTTP' as 'HTTP' | 'WEBSOCKET',
		endpoint: '',
		projectId: '',
		nodeId: '',
		topic: '',
		apiKey: '',
	})
	const federatedRoundForm = reactive({
		name: '',
		sampleCount: 1000,
	})
	const federatedUpdateForm = reactive({
		checkpointUri: '',
		sampleCount: 1000,
		notes: '',
	})
	const federatedAggregateForm = reactive({
		globalModelVersion: '',
		checkpointUri: '',
		summary: '',
	})

	let refreshTimer: number | undefined
	let streamSocket: WebSocket | undefined

	const selectedPipeline = computed(() =>
		pipelines.value.find(pipeline => pipeline.id === selectedPipelineId.value),
	)
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
	const activeResearchModel = computed(() =>
		researchModels.value.find(model => selectedPipeline.value?.activeModelId === model.id),
	)
	const activeDbModel = computed(() =>
		models.value.find(model => selectedPipeline.value?.activeModelId === model.id),
	)
	const activeModelLabel = computed(() => {
		if (!selectedPipeline.value?.activeModelId) return 'No model selected'
		return activeResearchModel.value?.run ?? activeDbModel.value?.name ?? selectedPipeline.value.activeModelId
	})
	const latestTrainingRun = computed(() => trainingRuns.value[0] ?? null)
	const processingModelCount = computed(() => researchModels.value.length + models.value.length)
	const linkedDataLake = computed(() =>
		dataLakes.value.find(lake => lake.id === selectedPipeline.value?.dataLakeConnectionId) ?? null,
	)
	const selectedExportAdapter = computed(() =>
		exportAdapters.value.find(adapter => adapter.type === exportForm.adapterType) ?? null,
	)
	const exportCredentialHint = computed(() => {
		switch (exportForm.adapterType) {
			case 'huggingface':
				return 'Token can be saved and reused for later exports.'
			case 'postgres':
				return 'Host, port, username, password, and database can be saved securely.'
			case 'mongodb':
				return 'URI and database can be saved securely.'
			default:
				return 'This adapter does not require reusable credentials.'
		}
	})
	const exportSettingsPreview = computed(() => {
		switch (exportForm.adapterType) {
			case 'json':
				return exportSettingsForm.jsonPrettyPrint ? 'Pretty JSON file in object storage' : 'Compact JSON file in object storage'
			case 'csv':
				return `CSV file with "${exportSettingsForm.csvDelimiter}" delimiter`
			case 'postgres':
				return `${exportSettingsForm.postgresHost || 'host'} / ${exportSettingsForm.postgresDatabase || 'database'} / ${exportSettingsForm.postgresTable || 'table'}`
			case 'mongodb':
				return `${exportSettingsForm.mongodbDatabase || 'database'} / ${exportSettingsForm.mongodbCollection || 'collection'}`
			case 'huggingface':
				return exportSettingsForm.huggingFaceRepoName
					? `datasets/${exportSettingsForm.huggingFaceRepoName}`
					: 'Dataset repo on Hugging Face'
			default:
				return 'Configure adapter settings'
		}
	})
	const supportsSavedCredentials = computed(() =>
		['huggingface', 'postgres', 'mongodb'].includes(exportForm.adapterType),
	)
	const federatedConfig = computed<FederatedConnectionState>(() => {
		const raw = (selectedPipeline.value?.streamConfig?.federated ?? {}) as Partial<FederatedConnectionState>
		return {
			enabled: false,
			protocol: 'HTTP',
			endpoint: '',
			projectId: null,
			nodeId: null,
			topic: null,
			headers: {},
			registerPayload: {},
			status: 'DISCONNECTED',
			lastConnectedAt: null,
			lastTestedAt: null,
			lastError: null,
			lastTestResult: null,
			registrationId: null,
			registeredAt: null,
			lastDeliveryAt: null,
			globalModelVersion: null,
			currentRoundId: null,
			rounds: [],
			...raw,
		}
	})
	const activeFederatedRound = computed(
		() => federatedRounds.value.find(round => round.id === federatedConfig.value.currentRoundId) ?? null,
	)
	const sourceMixSummary = computed(() => {
		const external = sources.value.filter(source => source.mode === 'EXTERNAL').length
		const simulated = sources.value.length - external
		return `${external} external / ${simulated} simulated`
	})
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
			applyAdapterDefaults(adapterType)
		},
		{ immediate: true },
	)

	watch(
		() => exportForm.stage,
		stage => {
			syncStageDependentExportFields(stage)
		},
	)

	onMounted(async () => {
		if (!federatedForm.endpoint) federatedForm.endpoint = demoFederatedEndpoint
		try {
			const presetResponse = await apiClient.getSimulatorPresets()
			simulatorPresets.value = presetResponse.data ?? null
		} catch {
			simulatorPresets.value = null
		}
		await loadPipelines()
		refreshTimer = window.setInterval(refreshEvents, 5000)
	})

	onUnmounted(() => {
		if (refreshTimer) window.clearInterval(refreshTimer)
		streamSocket?.close()
	})

	async function loadPipelines() {
		loading.value = true
		try {
			const response = await apiClient.listSmartCityPipelines()
			const adapterResponse = await apiClient.getExportAdapters()
			exportAdapters.value = adapterResponse.data ?? []
			if (!exportForm.targetName) exportForm.targetName = 'Business export'
			pipelines.value = response.data ?? []
			if (!selectedPipelineId.value) selectedPipelineId.value = pipelines.value[0]?.id ?? ''
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
			const lakeObjectsResponse = await apiClient.listSmartCityDataLakeObjects(pipelineId)
			dataLakeObjects.value = lakeObjectsResponse.data?.objects ?? []
			Object.assign(streamConfig, dashboard.pipeline.streamConfig)
			syncFederatedForm(dashboard.pipeline)
			await loadFederatedRounds(pipelineId)
			await loadObservability(pipelineId)
			await loadN8nWorkflow()
			await loadModels()
			connectWebSocket(pipelineId)
			addLog(`[SYNC] ${dashboard.sources.length} sources, ${dashboard.events.length} recent events`)
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load dashboard')
		} finally {
			loading.value = false
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

	async function loadFederatedRounds(pipelineId: string) {
		const response = await apiClient.listFederatedRounds(pipelineId)
		federatedRounds.value = response.data?.rounds ?? []
	}

	async function loadObservability(pipelineId: string) {
		const response = await apiClient.getSmartCityObservability(pipelineId)
		observability.value = response.data ?? null
	}

	function connectWebSocket(pipelineId: string) {
		streamSocket?.close()
		const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000'
		const wsUrl = `${apiUrl.replace(/^http/, 'ws')}/api/v1/smart-city/pipelines/${pipelineId}/ws`
		wsStatus.value = 'connecting'
		streamSocket = new WebSocket(wsUrl)
		streamSocket.onopen = () => {
			wsStatus.value = 'connected'
			addLog('[WS] Live stream connected')
		}
		streamSocket.onmessage = message => {
			const payload = JSON.parse(message.data)
			if (payload.type === 'snapshot') {
				events.value = payload.data.events ?? events.value
				return
			}
			if (payload.type === 'sensor_event') {
				events.value = [payload.data, ...events.value].slice(0, 50)
			}
			if (payload.type === 'processing_result') {
				latestProcessingResult.value = payload.data
				addLog(`[PROCESS] ${payload.data?.output?.kind ?? 'model'} produced result`)
			}
			if (payload.type === 'backfill_completed') {
				latestBackfill.value = payload.data as SmartCityBackfillResult
				addLog(`[BACKFILL] ${payload.data?.scannedEvents ?? 0} events replayed`)
			}
			if (payload.type === 'federated_status') {
				addLog(`[FEDERATED] ${payload.data?.status ?? 'status update'}`)
			}
		}
		streamSocket.onclose = () => {
			wsStatus.value = 'disconnected'
			addLog('[WS] Live stream disconnected')
		}
		streamSocket.onerror = () => {
			wsStatus.value = 'disconnected'
		}
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

	async function saveConfiguration() {
		if (!selectedPipelineId.value) return
		const response = await apiClient.updateSmartCityPipeline(selectedPipelineId.value, {
			streamConfig: { ...streamConfig },
		})
		replacePipeline(response.data)
		showConfigModal.value = false
		addLog(`[CONFIG] Saved ${streamConfig.throughputLimit} GB/s limit`)
		toast.success('Configuration saved')
	}

	async function saveDataLake() {
		if (!dataLakeForm.name.trim() || !dataLakeForm.bucket.trim()) {
			return toast.error('Data lake name and bucket are required')
		}
		saving.value = true
		try {
			const response = await apiClient.createDataLake({
				...dataLakeForm,
				region: dataLakeForm.region || undefined,
				endpoint: dataLakeForm.endpoint || undefined,
				basePrefix: dataLakeForm.basePrefix || undefined,
				accessKey: dataLakeForm.accessKey || undefined,
				secretKey: dataLakeForm.secretKey || undefined,
			})
			if (response.data) {
				dataLakes.value.unshift(response.data)
				if (selectedPipelineId.value) {
					const pipeline = await apiClient.updateSmartCityPipeline(selectedPipelineId.value, {
						dataLakeConnectionId: response.data.id,
					})
					replacePipeline(pipeline.data)
				}
				Object.assign(dataLakeForm, {
					name: '',
					provider: 'CUSTOM_S3',
					bucket: '',
					region: '',
					endpoint: '',
					basePrefix: '',
					accessKey: '',
					secretKey: '',
					isDefault: true,
				})
				showDataLakeModal.value = false
				addLog(`[DATALAKE] Connected ${response.data.bucket}`)
				toast.success('Data lake saved')
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not save data lake')
		} finally {
			saving.value = false
		}
	}

	async function testDataLake(lake: DataLakeConnection) {
		try {
			const response = await apiClient.testDataLake(lake.id)
			replaceDataLake(response.data)
			if (selectedPipelineId.value) {
				const lakeObjectsResponse = await apiClient.listSmartCityDataLakeObjects(selectedPipelineId.value)
				dataLakeObjects.value = lakeObjectsResponse.data?.objects ?? []
			}
			addLog(`[DATALAKE] Tested ${lake.bucket}`)
			toast.success('Data lake config is valid')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not test data lake')
		}
	}

	async function disconnectDataLake(lake: DataLakeConnection) {
		try {
			const response = await apiClient.disconnectDataLake(lake.id)
			replaceDataLake(response.data)
			addLog(`[DATALAKE] Disconnected ${lake.bucket}`)
			toast.success('Data lake disconnected')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not disconnect data lake')
		}
	}

	async function deleteDataLakeConnection(lake: DataLakeConnection) {
		if (!window.confirm(`Delete data lake "${lake.name}"?`)) return
		try {
			await apiClient.deleteDataLake(lake.id)
			dataLakes.value = dataLakes.value.filter(item => item.id !== lake.id)
			if (selectedPipeline.value?.dataLakeConnectionId === lake.id && selectedPipelineId.value) {
				const response = await apiClient.updateSmartCityPipeline(selectedPipelineId.value, {
					dataLakeConnectionId: null,
				})
				replacePipeline(response.data)
			}
			addLog(`[DATALAKE] Deleted ${lake.bucket}`)
			toast.success('Data lake deleted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not delete data lake')
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
		addLog('[PROCESS] Manual processing test completed')
	}

	async function exportStageToAdapter() {
		if (!selectedPipelineId.value) return
		saving.value = true
		try {
			const response = await apiClient.exportSmartCityPipelineStage(selectedPipelineId.value, {
				stage: exportForm.stage,
				adapterType: exportForm.adapterType,
				settings: buildExportSettings(),
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
		saving.value = true
		try {
			const response = await apiClient.createSmartCityExportTarget(selectedPipelineId.value, {
				name: exportForm.targetName,
				stage: exportForm.stage,
				adapterType: exportForm.adapterType,
				settings: buildExportSettings(),
				saveCredentials: exportForm.saveCredentials,
				isContinuous: exportForm.isContinuous,
				cadenceSeconds: exportForm.cadenceSeconds,
			})
			if (response.data) {
				exportTargets.value = [response.data, ...exportTargets.value]
				addLog(`[EXPORT] Saved target ${response.data.name}`)
				toast.success('Export target saved')
			}
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
			if (selectedPipelineId.value) {
				const runsResponse = await apiClient.listSmartCityExportRuns(selectedPipelineId.value)
				exportRuns.value = runsResponse.data ?? []
			}
			toast.success('Export target queued')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not queue export target')
		}
	}

	async function refreshLakeBrowser() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.listSmartCityDataLakeObjects(selectedPipelineId.value)
			dataLakeObjects.value = response.data?.objects ?? []
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load data lake objects')
		}
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

	function formatBytes(value: number) {
		if (value < 1024) return `${value} B`
		if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
		return `${(value / (1024 * 1024)).toFixed(1)} MB`
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

	async function connectFederated() {
		if (!selectedPipelineId.value) return
		if (!federatedForm.endpoint.trim()) return toast.error('Federated endpoint is required')
		saving.value = true
		try {
			const response = await apiClient.connectFederatedLearning(selectedPipelineId.value, {
				endpoint: federatedForm.endpoint,
				protocol: federatedForm.protocol,
				projectId: federatedForm.projectId || undefined,
				nodeId: federatedForm.nodeId || undefined,
				topic: federatedForm.topic || undefined,
				apiKey: federatedForm.apiKey || undefined,
			})
			if (response.data?.pipeline) {
				replacePipeline(response.data.pipeline)
				syncFederatedForm(response.data.pipeline)
			}
			await loadFederatedRounds(selectedPipelineId.value)
			addLog(`[FEDERATED] ${response.data?.connection.summary ?? 'Connected'}`)
			toast.success('Federated connection saved')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not connect federated learning')
		} finally {
			saving.value = false
		}
	}

	async function testFederatedConnection() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.testFederatedLearning(selectedPipelineId.value)
			addLog(`[FEDERATED] ${response.data?.summary ?? 'Connection test succeeded'}`)
			await loadDashboard(selectedPipelineId.value)
			toast.success('Federated connection tested')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not test federated connection')
		}
	}

	async function disconnectFederatedConnection() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.disconnectFederatedLearning(selectedPipelineId.value)
			replacePipeline(response.data)
			if (response.data) syncFederatedForm(response.data)
			federatedRounds.value = []
			addLog('[FEDERATED] Connection disabled')
			toast.success('Federated connection disconnected')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not disconnect federated connection')
		}
	}

	async function startFederatedRoundFlow() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.startFederatedRound(selectedPipelineId.value, {
				name: federatedRoundForm.name || undefined,
				sampleCount: federatedRoundForm.sampleCount,
			})
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(selectedPipelineId.value)
			addLog(`[FEDERATED] Started round ${response.data?.round.name ?? ''}`.trim())
			toast.success('Federated round started')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not start federated round')
		}
	}

	async function submitFederatedRoundUpdate() {
		if (!selectedPipelineId.value || !activeFederatedRound.value) return
		try {
			const response = await apiClient.submitFederatedUpdate(
				selectedPipelineId.value,
				activeFederatedRound.value.id,
				{
					checkpointUri: federatedUpdateForm.checkpointUri || undefined,
					sampleCount: federatedUpdateForm.sampleCount,
					notes: federatedUpdateForm.notes || undefined,
				},
			)
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(selectedPipelineId.value)
			addLog(`[FEDERATED] Submitted update for ${activeFederatedRound.value.name}`)
			toast.success('Round update submitted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not submit federated update')
		}
	}

	async function aggregateFederatedRoundFlow() {
		if (!selectedPipelineId.value || !activeFederatedRound.value) return
		try {
			const response = await apiClient.aggregateFederatedRound(
				selectedPipelineId.value,
				activeFederatedRound.value.id,
				{
					globalModelVersion: federatedAggregateForm.globalModelVersion || undefined,
					checkpointUri: federatedAggregateForm.checkpointUri || undefined,
					summary: federatedAggregateForm.summary || undefined,
				},
			)
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(selectedPipelineId.value)
			addLog(`[FEDERATED] Aggregated round ${activeFederatedRound.value.name}`)
			toast.success('Federated round aggregated')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not aggregate federated round')
		}
	}

	async function syncFederatedGlobalState() {
		if (!selectedPipelineId.value) return
		try {
			const response = await apiClient.syncFederatedGlobalModel(selectedPipelineId.value, {
				includeRounds: true,
			})
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			federatedRounds.value = response.data?.rounds ?? federatedRounds.value
			addLog(`[FEDERATED] ${response.data?.summary ?? 'Global state synced'}`)
			toast.success('Federated global state synced')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not sync global model')
		}
	}

	function openFederatedWorkspace() {
		viewMode.value = 'workflow'
		addLog('[FEDERATED] Opened orchestration view')
	}

	function syncFederatedForm(pipeline?: SmartCityPipeline) {
		const federated = (pipeline?.streamConfig?.federated ?? {}) as Partial<FederatedConnectionState>
		Object.assign(federatedForm, {
			protocol: federated.protocol ?? 'HTTP',
			endpoint: federated.endpoint ?? '',
			projectId: federated.projectId ?? '',
			nodeId: federated.nodeId ?? '',
			topic: federated.topic ?? '',
			apiKey: '',
		})
	}

	function replacePipeline(pipeline?: SmartCityPipeline) {
		if (!pipeline) return
		const index = pipelines.value.findIndex(item => item.id === pipeline.id)
		if (index >= 0) pipelines.value[index] = pipeline
	}

	function replaceSource(source?: SensorSource) {
		if (!source) return
		const index = sources.value.findIndex(item => item.id === source.id)
		if (index >= 0) sources.value[index] = source
	}

	function replaceDataLake(lake?: DataLakeConnection) {
		if (!lake) return
		const index = dataLakes.value.findIndex(item => item.id === lake.id)
		if (index >= 0) dataLakes.value[index] = lake
	}

	function replaceModel(model?: ModelArtifact) {
		if (!model) return
		const index = models.value.findIndex(item => item.id === model.id)
		if (index >= 0) models.value[index] = model
	}

	function replaceExportTarget(target?: SmartCityExportTarget) {
		if (!target) return
		const index = exportTargets.value.findIndex(item => item.id === target.id)
		if (index >= 0) exportTargets.value[index] = target
	}

	function applyAdapterDefaults(adapterType: ExportAdapterKind) {
		switch (adapterType) {
			case 'json':
				exportSettingsForm.jsonPrettyPrint = true
				break
			case 'csv':
				if (!exportSettingsForm.csvDelimiter) exportSettingsForm.csvDelimiter = ','
				break
			case 'postgres':
				if (!exportSettingsForm.postgresPort) exportSettingsForm.postgresPort = 5432
				if (!exportSettingsForm.postgresTable) exportSettingsForm.postgresTable = `smart_city_${exportForm.stage}`
				break
			case 'mongodb':
				if (!exportSettingsForm.mongodbCollection) {
					exportSettingsForm.mongodbCollection = `smart_city_${exportForm.stage}`
				}
				break
			case 'huggingface':
				if (!exportSettingsForm.huggingFaceFileName) {
					exportSettingsForm.huggingFaceFileName = `smart_city_${exportForm.stage}.csv`
				}
				break
		}
	}

	function syncStageDependentExportFields(stage: ExportStage) {
		const tableName = `smart_city_${stage}`
		const fileName = `smart_city_${stage}.csv`
		if (!exportSettingsForm.postgresTable || exportSettingsForm.postgresTable.startsWith('smart_city_')) {
			exportSettingsForm.postgresTable = tableName
		}
		if (
			!exportSettingsForm.mongodbCollection ||
			exportSettingsForm.mongodbCollection.startsWith('smart_city_')
		) {
			exportSettingsForm.mongodbCollection = tableName
		}
		if (
			!exportSettingsForm.huggingFaceFileName ||
			exportSettingsForm.huggingFaceFileName.startsWith('smart_city_')
		) {
			exportSettingsForm.huggingFaceFileName = fileName
		}
	}

	function buildExportSettings(): Record<string, unknown> {
		const settings = buildStructuredExportSettings()
		const advanced = parseAdvancedExportSettings()
		return { ...settings, ...advanced }
	}

	function buildStructuredExportSettings(): Record<string, unknown> {
		switch (exportForm.adapterType) {
			case 'json':
				return {
					prettyPrint: exportSettingsForm.jsonPrettyPrint,
				}
			case 'csv':
				return {
					delimiter: exportSettingsForm.csvDelimiter || ',',
				}
			case 'postgres':
				return {
					host: exportSettingsForm.postgresHost.trim(),
					port: Number(exportSettingsForm.postgresPort),
					username: exportSettingsForm.postgresUsername.trim(),
					password: exportSettingsForm.postgresPassword,
					database: exportSettingsForm.postgresDatabase.trim(),
					table: exportSettingsForm.postgresTable.trim(),
					ifExists: exportSettingsForm.postgresIfExists,
				}
			case 'mongodb':
				return {
					uri: exportSettingsForm.mongodbUri.trim(),
					database: exportSettingsForm.mongodbDatabase.trim(),
					collection: exportSettingsForm.mongodbCollection.trim(),
					ifExists: exportSettingsForm.mongodbIfExists,
				}
			case 'huggingface':
				return {
					token: exportSettingsForm.huggingFaceToken.trim(),
					repoName: exportSettingsForm.huggingFaceRepoName.trim(),
					fileName: exportSettingsForm.huggingFaceFileName.trim(),
					commitMessage: exportSettingsForm.huggingFaceCommitMessage.trim(),
					private: exportSettingsForm.huggingFacePrivate,
				}
			default:
				return {}
		}
	}

	function parseAdvancedExportSettings(): Record<string, unknown> {
		if (!exportSettingsForm.advancedJson.trim()) return {}
		return JSON.parse(exportSettingsForm.advancedJson) as Record<string, unknown>
	}

	function loadExportTargetIntoForm(target: SmartCityExportTarget) {
		exportForm.targetName = target.name
		exportForm.stage = target.stage
		exportForm.adapterType = target.adapterType as ExportAdapterKind
		exportForm.saveCredentials = target.saveCredentials
		exportForm.isContinuous = target.isContinuous
		exportForm.cadenceSeconds = target.cadenceSeconds
		applyExportSettingsToForm(target.adapterType as ExportAdapterKind, target.settingsJson)
		addLog(`[EXPORT] Loaded target ${target.name} into form`)
		toast.success('Export target loaded into form')
	}

	function applyExportSettingsToForm(
		adapterType: ExportAdapterKind,
		settings: Record<string, unknown> = {},
	) {
		exportSettingsForm.advancedJson = ''
		switch (adapterType) {
			case 'json':
				exportSettingsForm.jsonPrettyPrint = Boolean(settings.prettyPrint ?? true)
				break
			case 'csv':
				exportSettingsForm.csvDelimiter = String(settings.delimiter ?? ',')
				break
			case 'postgres':
				exportSettingsForm.postgresHost = String(settings.host ?? '')
				exportSettingsForm.postgresPort = Number(settings.port ?? 5432)
				exportSettingsForm.postgresUsername = String(settings.username ?? '')
				exportSettingsForm.postgresPassword = String(settings.password ?? '')
				exportSettingsForm.postgresDatabase = String(settings.database ?? '')
				exportSettingsForm.postgresTable = String(settings.table ?? `smart_city_${exportForm.stage}`)
				exportSettingsForm.postgresIfExists =
					settings.ifExists === 'replace' ? 'replace' : 'append'
				break
			case 'mongodb':
				exportSettingsForm.mongodbUri = String(settings.uri ?? '')
				exportSettingsForm.mongodbDatabase = String(settings.database ?? '')
				exportSettingsForm.mongodbCollection = String(
					settings.collection ?? `smart_city_${exportForm.stage}`,
				)
				exportSettingsForm.mongodbIfExists =
					settings.ifExists === 'replace' ? 'replace' : 'append'
				break
			case 'huggingface':
				exportSettingsForm.huggingFaceToken = String(settings.token ?? '')
				exportSettingsForm.huggingFaceRepoName = String(settings.repoName ?? '')
				exportSettingsForm.huggingFaceFileName = String(
					settings.fileName ?? `smart_city_${exportForm.stage}.csv`,
				)
				exportSettingsForm.huggingFaceCommitMessage = String(settings.commitMessage ?? '')
				exportSettingsForm.huggingFacePrivate = Boolean(settings.private ?? false)
				break
		}
	}

	function addLog(message: string) {
		logs.value.unshift(`${new Date().toLocaleTimeString()} ${message}`)
		logs.value = logs.value.slice(0, 80)
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

	function sourceIcon(kind: string) {
		if (kind === 'video') return Video
		if (kind === 'power') return Zap
		if (kind === 'network') return Router
		return Wifi
	}

	function formatDateTime(value?: string | null) {
		if (!value) return 'Not available'
		return new Date(value).toLocaleString()
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
		dataLakeObjects,
		latestBackfill,
		federatedRounds,
		observability,
		models,
		trainingRuns,
		researchModels,
		latestProcessingResult,
		n8nWorkflow,
		logs,
		wsStatus,
		simulatorPresets,
		demoFederatedEndpoint,
		demoSimulatorBaseUrl,
		showPipelineModal,
		showSourceModal,
		showConfigModal,
		showDataLakeModal,
		showAdvancedExportJson,
		pipelineForm,
		sourceForm,
		dataLakeForm,
		exportForm,
		backfillForm,
		exportSettingsForm,
		streamConfig,
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
		activeResearchModel,
		activeDbModel,
		activeModelLabel,
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
		saveConfiguration,
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
		runExportTarget,
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
