import type { Ref } from 'vue'
import type { FederatedConnectionState, SmartCityPipeline } from '@/api/client'

export function buildFederatedConfig(
	pipeline: SmartCityPipeline | undefined,
): FederatedConnectionState {
	const raw = (pipeline?.streamConfig?.federated ?? {}) as Partial<FederatedConnectionState>
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
}

export function syncFederatedFormState(
	form: {
		protocol: 'HTTP' | 'WEBSOCKET'
		endpoint: string
		projectId: string
		nodeId: string
		topic: string
		apiKey: string
	},
	pipeline?: SmartCityPipeline,
): void {
	const federated = (pipeline?.streamConfig?.federated ?? {}) as Partial<FederatedConnectionState>
	Object.assign(form, {
		protocol: federated.protocol ?? 'HTTP',
		endpoint: federated.endpoint ?? '',
		projectId: federated.projectId ?? '',
		nodeId: federated.nodeId ?? '',
		topic: federated.topic ?? '',
		apiKey: '',
	})
}

export interface PipelineLiveStreamRefs {
	events: Ref<import('@/api/client').SensorEvent[]>
	latestProcessingResult: Ref<import('./pipeline-workbench.types').ProcessingTestResult | null>
	processingError: Ref<string | null>
	latestBackfill: Ref<import('@/api/client').SmartCityBackfillResult | null>
	wsStatus: Ref<'disconnected' | 'connecting' | 'connected'>
}

export function createPipelineLiveStream(
	refs: PipelineLiveStreamRefs,
	addLog: (message: string) => void,
) {
	let streamSocket: WebSocket | undefined

	function connect(pipelineId: string) {
		streamSocket?.close()
		const configured = (import.meta.env.VITE_API_URL || `${window.location.origin}/api/v1`).replace(/\/$/, '')
		const httpBase = configured.startsWith('http')
			? configured
			: `${window.location.origin}${configured.startsWith('/') ? configured : `/${configured}`}`
		const wsUrl = `${httpBase.replace(/^http/, 'ws')}/smart-city/pipelines/${pipelineId}/ws`
		refs.wsStatus.value = 'connecting'
		streamSocket = new WebSocket(wsUrl)
		streamSocket.onopen = () => {
			refs.wsStatus.value = 'connected'
			addLog('[WS] Live stream connected')
		}
		streamSocket.onmessage = message => {
			const payload = JSON.parse(message.data)
			if (payload.type === 'snapshot') {
				refs.events.value = payload.data.events ?? refs.events.value
				return
			}
			if (payload.type === 'sensor_event') {
				refs.events.value = [payload.data, ...refs.events.value].slice(0, 50)
			}
			if (payload.type === 'processing_result') {
				if (payload.data?.error) return
				refs.latestProcessingResult.value = payload.data
				refs.processingError.value = null
				addLog(`[PROCESS] ${payload.data?.output?.kind ?? 'model'} produced result`)
			}
			if (payload.type === 'processing_error') {
				refs.processingError.value =
					typeof payload.data?.error === 'string' ? payload.data.error : 'Model inference failed'
			}
			if (payload.type === 'backfill_completed') {
				refs.latestBackfill.value = payload.data
				addLog(`[BACKFILL] ${payload.data?.scannedEvents ?? 0} events replayed`)
			}
			if (payload.type === 'federated_status') {
				addLog(`[FEDERATED] ${payload.data?.status ?? 'status update'}`)
			}
		}
		streamSocket.onclose = () => {
			refs.wsStatus.value = 'disconnected'
			addLog('[WS] Live stream disconnected')
		}
		streamSocket.onerror = () => {
			refs.wsStatus.value = 'disconnected'
		}
	}

	function disconnect() {
		streamSocket?.close()
		streamSocket = undefined
	}

	return { connect, disconnect }
}
