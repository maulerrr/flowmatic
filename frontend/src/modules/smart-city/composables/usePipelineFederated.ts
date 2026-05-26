import { computed, reactive, type ComputedRef, type Ref } from 'vue'
import { toast } from 'vue-sonner'
import {
	apiClient,
	type FederatedRound,
	type SmartCityPipeline,
} from '@/api/client'
import { replaceInList } from './pipeline-entity.helpers'
import { buildFederatedConfig, syncFederatedFormState } from './pipeline-runtime.helpers'

interface UsePipelineFederatedOptions {
	selectedPipelineId: Ref<string>
	selectedPipeline: ComputedRef<SmartCityPipeline | undefined>
	pipelines: Ref<SmartCityPipeline[]>
	federatedRounds: Ref<FederatedRound[]>
	saving: Ref<boolean>
	viewMode: Ref<'dashboard' | 'workflow'>
	demoFederatedEndpoint: string
	addLog: (message: string) => void
	loadDashboard: (pipelineId: string) => Promise<void>
}

export function usePipelineFederated(options: UsePipelineFederatedOptions) {
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

	const federatedConfig = computed(() => buildFederatedConfig(options.selectedPipeline.value))
	const activeFederatedRound = computed(
		() =>
			options.federatedRounds.value.find(round => round.id === federatedConfig.value.currentRoundId) ??
			null,
	)

	function replacePipeline(pipeline?: SmartCityPipeline) {
		replaceInList(options.pipelines.value, pipeline)
	}

	function syncFederatedForm(pipeline?: SmartCityPipeline) {
		syncFederatedFormState(federatedForm, pipeline)
	}

	async function loadFederatedRounds(pipelineId: string) {
		const response = await apiClient.listFederatedRounds(pipelineId)
		options.federatedRounds.value = response.data?.rounds ?? []
	}

	async function connectFederated() {
		if (!options.selectedPipelineId.value) return
		if (!federatedForm.endpoint.trim()) return toast.error('Federated endpoint is required')
		options.saving.value = true
		try {
			const response = await apiClient.connectFederatedLearning(options.selectedPipelineId.value, {
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
			await loadFederatedRounds(options.selectedPipelineId.value)
			options.addLog(`[FEDERATED] ${response.data?.connection.summary ?? 'Connected'}`)
			toast.success('Federated connection saved')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not connect federated learning')
		} finally {
			options.saving.value = false
		}
	}

	async function testFederatedConnection() {
		if (!options.selectedPipelineId.value) return
		try {
			const response = await apiClient.testFederatedLearning(options.selectedPipelineId.value)
			options.addLog(`[FEDERATED] ${response.data?.summary ?? 'Connection test succeeded'}`)
			await options.loadDashboard(options.selectedPipelineId.value)
			toast.success('Federated connection tested')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not test federated connection')
		}
	}

	async function disconnectFederatedConnection() {
		if (!options.selectedPipelineId.value) return
		try {
			const response = await apiClient.disconnectFederatedLearning(options.selectedPipelineId.value)
			replacePipeline(response.data)
			if (response.data) syncFederatedForm(response.data)
			options.federatedRounds.value = []
			options.addLog('[FEDERATED] Connection disabled')
			toast.success('Federated connection disconnected')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not disconnect federated connection')
		}
	}

	async function startFederatedRoundFlow() {
		if (!options.selectedPipelineId.value) return
		try {
			const response = await apiClient.startFederatedRound(options.selectedPipelineId.value, {
				name: federatedRoundForm.name || undefined,
				sampleCount: federatedRoundForm.sampleCount,
			})
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(options.selectedPipelineId.value)
			options.addLog(`[FEDERATED] Started round ${response.data?.round.name ?? ''}`.trim())
			toast.success('Federated round started')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not start federated round')
		}
	}

	async function submitFederatedRoundUpdate() {
		if (!options.selectedPipelineId.value || !activeFederatedRound.value) return
		try {
			const response = await apiClient.submitFederatedUpdate(
				options.selectedPipelineId.value,
				activeFederatedRound.value.id,
				{
					checkpointUri: federatedUpdateForm.checkpointUri || undefined,
					sampleCount: federatedUpdateForm.sampleCount,
					notes: federatedUpdateForm.notes || undefined,
				},
			)
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(options.selectedPipelineId.value)
			options.addLog(`[FEDERATED] Submitted update for ${activeFederatedRound.value.name}`)
			toast.success('Round update submitted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not submit federated update')
		}
	}

	async function aggregateFederatedRoundFlow() {
		if (!options.selectedPipelineId.value || !activeFederatedRound.value) return
		try {
			const response = await apiClient.aggregateFederatedRound(
				options.selectedPipelineId.value,
				activeFederatedRound.value.id,
				{
					globalModelVersion: federatedAggregateForm.globalModelVersion || undefined,
					checkpointUri: federatedAggregateForm.checkpointUri || undefined,
					summary: federatedAggregateForm.summary || undefined,
				},
			)
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			await loadFederatedRounds(options.selectedPipelineId.value)
			options.addLog(`[FEDERATED] Aggregated round ${activeFederatedRound.value.name}`)
			toast.success('Federated round aggregated')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not aggregate federated round')
		}
	}

	async function syncFederatedGlobalState() {
		if (!options.selectedPipelineId.value) return
		try {
			const response = await apiClient.syncFederatedGlobalModel(options.selectedPipelineId.value, {
				includeRounds: true,
			})
			if (response.data?.pipeline) replacePipeline(response.data.pipeline)
			options.federatedRounds.value = response.data?.rounds ?? options.federatedRounds.value
			options.addLog(`[FEDERATED] ${response.data?.summary ?? 'Global state synced'}`)
			toast.success('Federated global state synced')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not sync global model')
		}
	}

	function openFederatedWorkspace() {
		options.viewMode.value = 'workflow'
		options.addLog('[FEDERATED] Opened orchestration view')
	}

	function bootstrapFederatedEndpoint() {
		if (!federatedForm.endpoint) federatedForm.endpoint = options.demoFederatedEndpoint
	}

	return {
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
		bootstrapFederatedEndpoint,
	}
}
