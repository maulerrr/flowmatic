import { reactive, type ComputedRef, type Ref } from 'vue'
import { toast } from 'vue-sonner'
import {
	apiClient,
	type DataLakeConnection,
	type DataLakeObjectGroup,
	type SmartCityPipeline,
} from '@/api/client'
import { replaceInList } from './pipeline-entity.helpers'

interface UsePipelineDataLakeOptions {
	selectedPipelineId: Ref<string>
	selectedPipeline: ComputedRef<SmartCityPipeline | undefined>
	pipelines: Ref<SmartCityPipeline[]>
	dataLakes: Ref<DataLakeConnection[]>
	dataLakeObjects: Ref<DataLakeObjectGroup[]>
	saving: Ref<boolean>
	showDataLakeModal: Ref<boolean>
	addLog: (message: string) => void
}

export function usePipelineDataLake(options: UsePipelineDataLakeOptions) {
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

	function replacePipeline(pipeline?: SmartCityPipeline) {
		replaceInList(options.pipelines.value, pipeline)
	}

	function replaceDataLake(lake?: DataLakeConnection) {
		replaceInList(options.dataLakes.value, lake)
	}

	async function saveDataLake() {
		if (!dataLakeForm.name.trim() || !dataLakeForm.bucket.trim()) {
			return toast.error('Data lake name and bucket are required')
		}
		options.saving.value = true
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
				options.dataLakes.value.unshift(response.data)
				if (options.selectedPipelineId.value) {
					const pipeline = await apiClient.updateSmartCityPipeline(options.selectedPipelineId.value, {
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
				options.showDataLakeModal.value = false
				options.addLog(`[DATALAKE] Connected ${response.data.bucket}`)
				toast.success('Data lake saved')
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not save data lake')
		} finally {
			options.saving.value = false
		}
	}

	async function testDataLake(lake: DataLakeConnection) {
		try {
			const response = await apiClient.testDataLake(lake.id)
			replaceDataLake(response.data)
			if (options.selectedPipelineId.value) {
				const lakeObjectsResponse = await apiClient.listSmartCityDataLakeObjects(
					options.selectedPipelineId.value,
				)
				options.dataLakeObjects.value = lakeObjectsResponse.data?.objects ?? []
			}
			options.addLog(`[DATALAKE] Tested ${lake.bucket}`)
			toast.success('Data lake config is valid')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not test data lake')
		}
	}

	async function disconnectDataLake(lake: DataLakeConnection) {
		try {
			const response = await apiClient.disconnectDataLake(lake.id)
			replaceDataLake(response.data)
			options.addLog(`[DATALAKE] Disconnected ${lake.bucket}`)
			toast.success('Data lake disconnected')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not disconnect data lake')
		}
	}

	async function deleteDataLakeConnection(lake: DataLakeConnection) {
		if (!window.confirm(`Delete data lake "${lake.name}"?`)) return
		try {
			await apiClient.deleteDataLake(lake.id)
			options.dataLakes.value = options.dataLakes.value.filter(item => item.id !== lake.id)
			if (
				options.selectedPipeline.value?.dataLakeConnectionId === lake.id &&
				options.selectedPipelineId.value
			) {
				const response = await apiClient.updateSmartCityPipeline(options.selectedPipelineId.value, {
					dataLakeConnectionId: null,
				})
				replacePipeline(response.data)
			}
			options.addLog(`[DATALAKE] Deleted ${lake.bucket}`)
			toast.success('Data lake deleted')
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not delete data lake')
		}
	}

	async function refreshLakeBrowser() {
		if (!options.selectedPipelineId.value) return
		try {
			const response = await apiClient.listSmartCityDataLakeObjects(options.selectedPipelineId.value)
			options.dataLakeObjects.value = response.data?.objects ?? []
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load data lake objects')
		}
	}

	return {
		dataLakeForm,
		saveDataLake,
		testDataLake,
		disconnectDataLake,
		deleteDataLakeConnection,
		refreshLakeBrowser,
		replaceDataLake,
	}
}
