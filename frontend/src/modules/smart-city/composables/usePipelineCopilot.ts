import { ref } from 'vue'
import { toast } from 'vue-sonner'
import {
	apiClient,
	type CopilotChip,
	type CopilotVizCommand,
	type PipelineCopilotChatTurn,
	type PipelineCopilotContext,
} from '@/api/client'

export function usePipelineCopilot(getPipelineId: () => string) {
	const loading = ref(false)
	const context = ref<PipelineCopilotContext | null>(null)
	const messages = ref<PipelineCopilotChatTurn[]>([])
	const draft = ref('')

	function slashCommandForChip(chip: CopilotChip): string | null {
		const command = context.value?.vizCommands.find(item => item.id === chip.id)
		return command ? `/${command.slash}` : null
	}

	async function loadContext() {
		const pipelineId = getPipelineId()
		if (!pipelineId) {
			context.value = null
			messages.value = []
			return
		}
		loading.value = true
		try {
			const [contextResponse, historyResponse] = await Promise.all([
				apiClient.getPipelineCopilotContext(pipelineId),
				apiClient.getPipelineCopilotHistory(pipelineId),
			])
			context.value = contextResponse.data ?? null
			messages.value = historyResponse.data ?? []
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not load pipeline insights')
		} finally {
			loading.value = false
		}
	}

	async function postMessage(outgoing: string, displayText?: string): Promise<PipelineCopilotChatTurn | undefined> {
		const pipelineId = getPipelineId()
		if (!pipelineId || !outgoing.trim()) return undefined

		messages.value.push({
			id: `user-${Date.now()}`,
			role: 'user',
			text: displayText ?? outgoing,
		})
		loading.value = true
		try {
			const response = await apiClient.postPipelineCopilotChat(pipelineId, { message: outgoing })
			if (response.data) {
				context.value = response.data.context
				messages.value.push(response.data.turn)
				return response.data.turn
			}
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not analyze pipeline')
		} finally {
			loading.value = false
		}
		return undefined
	}

	async function activateChip(chip: CopilotChip): Promise<PipelineCopilotChatTurn | undefined> {
		const slash = slashCommandForChip(chip)
		if (!slash) return undefined
		return postMessage(slash, slash)
	}

	async function sendMessage() {
		const message = draft.value.trim()
		if (!message) return
		draft.value = ''
		await postMessage(message)
	}

	async function resetConversation() {
		const pipelineId = getPipelineId()
		if (!pipelineId) return
		loading.value = true
		try {
			await apiClient.resetPipelineCopilotSession(pipelineId)
			messages.value = []
			await loadContext()
		} catch (error) {
			toast.error(error instanceof Error ? error.message : 'Could not reset conversation')
		} finally {
			loading.value = false
		}
	}

	return {
		loading,
		context,
		messages,
		draft,
		loadContext,
		activateChip,
		sendMessage,
		resetConversation,
	}
}

export function scoreVizCommand(command: CopilotVizCommand, query: string): number {
	const normalized = query.toLowerCase().trim()
	if (!normalized) return 0

	const slash = command.slash.toLowerCase()
	const label = command.label.toLowerCase()
	const description = command.description.toLowerCase()
	const slashStem = slash.replace(/-/g, '')
	const queryStem = normalized.replace(/-/g, '')

	if (slash === normalized) return 200
	if (slash.startsWith(normalized)) return 180 - slash.length
	if (slashStem.startsWith(queryStem)) return 170 - slashStem.length
	if (label.startsWith(normalized)) return 140
	if (label.split(/\s+/).some(word => word.startsWith(normalized))) return 130
	if (slash.includes(normalized)) return 110
	if (label.includes(normalized)) return 90
	if (description.includes(normalized)) return 70
	if (description.split(/\s+/).some(word => word.startsWith(normalized))) return 60

	return -1
}

export function filterVizCommands(commands: CopilotVizCommand[], query: string) {
	const normalized = query.toLowerCase().trim()
	if (!normalized) return commands

	return commands
		.map(command => ({ command, score: scoreVizCommand(command, normalized) }))
		.filter(entry => entry.score >= 0)
		.sort((left, right) => right.score - left.score || left.command.label.localeCompare(right.command.label))
		.map(entry => entry.command)
}
