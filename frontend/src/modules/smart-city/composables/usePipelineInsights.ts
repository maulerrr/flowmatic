import { onMounted, ref, watch } from 'vue'
import { toast } from 'vue-sonner'
import {
	apiClient,
	type PipelineInsightConfig,
	type PipelineInsightRun,
} from '@/api/client'

export function usePipelineInsights(getPipelineId: () => string) {
	const loading = ref(false)
	const running = ref(false)
	const config = ref<PipelineInsightConfig | null>(null)
	const runs = ref<PipelineInsightRun[]>([])
	const selectedRunId = ref('')

	async function loadAll() {
		const pipelineId = getPipelineId()
		if (!pipelineId) {
			config.value = null
			runs.value = []
			return
		}
		loading.value = true
		try {
			const [configResponse, runsResponse] = await Promise.all([
				apiClient.getPipelineInsightConfig(pipelineId),
				apiClient.listPipelineInsightRuns(pipelineId),
			])
			config.value = configResponse.data ?? null
			runs.value = runsResponse.data ?? []
			if (!selectedRunId.value && runs.value.length) {
				selectedRunId.value = runs.value[0].id
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load insight engine')
		} finally {
			loading.value = false
		}
	}

	async function saveConfig(input: Partial<Pick<PipelineInsightConfig, 'intervalMinutes' | 'depth' | 'focus'>>) {
		const pipelineId = getPipelineId()
		if (!pipelineId) return
		loading.value = true
		try {
			const response = await apiClient.updatePipelineInsightConfig(pipelineId, input)
			config.value = response.data ?? config.value
			toast.success('Insight schedule updated')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not update insight schedule')
		} finally {
			loading.value = false
		}
	}

	async function runNow() {
		const pipelineId = getPipelineId()
		if (!pipelineId) return
		running.value = true
		try {
			const response = await apiClient.triggerPipelineInsightRun(pipelineId)
			if (response.data) {
				runs.value = [response.data, ...runs.value.filter(run => run.id !== response.data!.id)]
				selectedRunId.value = response.data.id
				toast.success('Insight analysis completed')
			}
			await loadAll()
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Insight analysis failed')
		} finally {
			running.value = false
		}
	}

	watch(
		() => getPipelineId(),
		() => {
			selectedRunId.value = ''
			void loadAll()
		},
	)

	onMounted(() => {
		void loadAll()
	})

	return {
		loading,
		running,
		config,
		runs,
		selectedRunId,
		loadAll,
		saveConfig,
		runNow,
	}
}
