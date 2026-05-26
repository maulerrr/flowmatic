<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { Bot, BarChart3, MessageSquareText, RefreshCw, SendHorizontal, Sparkles } from 'lucide-vue-next'
import type { CopilotChip, CopilotVizCommand } from '@/api/client'
import { filterVizCommands, usePipelineCopilot } from '../composables/usePipelineCopilot'
import PipelineCopilotChart from './pipeline-copilot-chart.vue'

const props = defineProps<{
	pipelineId: string
	pipelineName: string
}>()

const copilot = usePipelineCopilot(() => props.pipelineId)
const { loading, messages, draft, activateChip, sendMessage, resetConversation } = copilot

const composerInput = ref<HTMLInputElement | null>(null)
const composerEl = ref<HTMLElement | null>(null)
const slashMenuEl = ref<HTMLElement | null>(null)
const slashMenuIndex = ref(0)
const composerCursor = ref(0)
const focusedChartId = ref<string | null>(null)
const chatComposerVisible = ref(true)

let composerObserver: IntersectionObserver | null = null

const primaryVisualizations = computed(() => copilot.context.value?.primaryVisualizations ?? [])
const chips = computed(() => copilot.context.value?.chips ?? [])
const vizCommands = computed(() => copilot.context.value?.vizCommands ?? [])

const composerPlaceholder = 'Ask a question, or type / for charts…'

const slashState = computed(() => {
	const value = draft.value
	const cursor = composerCursor.value
	const beforeCursor = value.slice(0, cursor)
	const slashIndex = beforeCursor.lastIndexOf('/')
	if (slashIndex === -1) return null
	const query = beforeCursor.slice(slashIndex + 1)
	if (/\s/.test(query)) return null
	return { slashIndex, query, cursor }
})

const showSlashMenu = computed(() => slashState.value !== null && vizCommands.value.length > 0)

const filteredSlashCommands = computed(() => {
	if (!slashState.value) return []
	return filterVizCommands(vizCommands.value, slashState.value.query)
})

watch(
	() => slashState.value?.query,
	() => {
		slashMenuIndex.value = 0
		scrollActiveSlashItemIntoView()
	},
)

watch(filteredSlashCommands, commands => {
	if (commands.length === 0) {
		slashMenuIndex.value = 0
		return
	}
	if (slashMenuIndex.value >= commands.length) {
		slashMenuIndex.value = commands.length - 1
	}
	scrollActiveSlashItemIntoView()
})

watch(slashMenuIndex, () => {
	scrollActiveSlashItemIntoView()
})

watch(
	() => props.pipelineId,
	() => {
		copilot.messages.value = []
		if (props.pipelineId) void copilot.loadContext()
	},
	{ immediate: true },
)

function setupComposerObserver() {
	composerObserver?.disconnect()
	composerObserver = null
	const target = composerEl.value
	if (!target) return

	composerObserver = new IntersectionObserver(
		([entry]) => {
			chatComposerVisible.value = entry.isIntersecting
		},
		{ threshold: 0.12, rootMargin: '0px 0px -12px 0px' },
	)
	composerObserver.observe(target)
}

onMounted(() => {
	setupComposerObserver()
})

onUnmounted(() => {
	composerObserver?.disconnect()
})

watch(composerEl, () => {
	setupComposerObserver()
})

async function jumpToChat() {
	composerEl.value?.scrollIntoView({ behavior: 'smooth', block: 'end' })
	await nextTick()
	composerInput.value?.focus({ preventScroll: true })
	syncComposerCursor()
}

function syncComposerCursor() {
	composerCursor.value = composerInput.value?.selectionStart ?? draft.value.length
}

function scrollActiveSlashItemIntoView() {
	void nextTick(() => {
		const menu = slashMenuEl.value
		if (!menu) return
		menu.querySelector('.pipeline-insights__slash-item--active')?.scrollIntoView({ block: 'nearest' })
	})
}

async function scrollToVisualization(turnId: string) {
	await nextTick()
	await nextTick()
	const chartEl = document.getElementById(`chat-viz-${turnId}`)
	if (!chartEl) return

	chartEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
	focusedChartId.value = turnId
	chartEl.querySelector<HTMLElement>('.copilot-chart')?.focus({ preventScroll: true })

	window.setTimeout(() => {
		if (focusedChartId.value === turnId) focusedChartId.value = null
	}, 2000)
}

async function onChipClick(chip: CopilotChip) {
	const turn = await activateChip(chip)
	if (turn?.visualization) await scrollToVisualization(turn.id)
}

function insertSlashCommand(command: CopilotVizCommand) {
	const state = slashState.value
	if (!state) return
	const before = draft.value.slice(0, state.slashIndex)
	const after = draft.value.slice(state.cursor)
	const insertion = `/${command.slash} `
	draft.value = `${before}${insertion}${after}`.replace(/\s+/g, ' ').replace(/^\s+/, '')
	slashMenuIndex.value = 0
	void nextTick(() => {
		const input = composerInput.value
		if (!input) return
		const nextCursor = before.length + insertion.length
		input.focus()
		input.setSelectionRange(nextCursor, nextCursor)
		syncComposerCursor()
	})
}

function onComposerInput() {
	syncComposerCursor()
}

function onComposerKeydown(event: KeyboardEvent) {
	syncComposerCursor()

	if (!showSlashMenu.value || filteredSlashCommands.value.length === 0) return

	if (event.key === 'ArrowDown') {
		event.preventDefault()
		slashMenuIndex.value = (slashMenuIndex.value + 1) % filteredSlashCommands.value.length
	} else if (event.key === 'ArrowUp') {
		event.preventDefault()
		slashMenuIndex.value =
			(slashMenuIndex.value - 1 + filteredSlashCommands.value.length) % filteredSlashCommands.value.length
	} else if (event.key === 'Enter' || event.key === 'Tab') {
		event.preventDefault()
		insertSlashCommand(filteredSlashCommands.value[slashMenuIndex.value])
	} else if (event.key === 'Escape') {
		event.preventDefault()
		draft.value = draft.value.replace(/\/[^\s]*$/, '')
		syncComposerCursor()
	}
}
</script>

<template>
	<section class="pipeline-insights">
		<div class="pipeline-insights__header">
			<div class="flex items-start gap-3">
				<div class="pipeline-insights__avatar">
					<Sparkles class="w-5 h-5" />
				</div>
				<div>
					<div class="pipeline-insights__eyebrow">Pipeline insights</div>
					<h2 class="pipeline-insights__title">Chat with {{ pipelineName }}</h2>
					<p class="pipeline-insights__summary">
						Ask in plain language. Type <code class="pipeline-insights__code">/</code> to pick a chart — combine both in one message.
					</p>
				</div>
			</div>
			<button type="button" class="pipeline-insights__refresh" @click="resetConversation()">
				<RefreshCw class="w-4 h-4" /> Refresh context
			</button>
		</div>

		<div v-if="primaryVisualizations.length" class="pipeline-insights__primary-grid">
			<PipelineCopilotChart
				v-for="visualization in primaryVisualizations"
				:key="visualization.id"
				:spec="visualization"
			/>
		</div>

		<div v-if="chips.length" class="pipeline-insights__chips">
			<button
				v-for="chip in chips"
				:key="chip.id"
				type="button"
				class="pipeline-insights__chip"
				:class="`pipeline-insights__chip--${chip.category}`"
				:disabled="loading"
				@click="onChipClick(chip)"
			>
				{{ chip.label }}
			</button>
		</div>

		<div class="pipeline-insights__chat">
			<div class="pipeline-insights__messages">
				<div v-if="!messages.length && !loading" class="pipeline-insights__empty">
					Start with a question, or type <code class="pipeline-insights__code">/</code> to draw a graph.
				</div>
				<div
					v-for="message in messages"
					:key="message.id"
					:class="[
						'pipeline-insights__message',
						message.role === 'assistant' ? 'pipeline-insights__message--assistant' : 'pipeline-insights__message--user',
					]"
				>
					<div v-if="message.role === 'assistant'" class="pipeline-insights__message-icon">
						<Bot class="w-4 h-4" />
					</div>
					<div class="pipeline-insights__message-body">
						<p v-if="message.text">{{ message.text }}</p>
						<div
							v-if="message.visualization"
							:id="`chat-viz-${message.id}`"
							class="pipeline-insights__inline-chart"
							:class="{ 'pipeline-insights__inline-chart--focus': focusedChartId === message.id }"
						>
							<PipelineCopilotChart :spec="message.visualization" />
						</div>
					</div>
				</div>
				<div v-if="loading" class="pipeline-insights__loading">
					{{ copilot.context.value?.llmEnabled ? 'Thinking…' : 'Working…' }}
				</div>
			</div>

			<form ref="composerEl" class="pipeline-insights__composer" @submit.prevent="sendMessage()">
				<div class="pipeline-insights__composer-shell">
					<div
						v-if="showSlashMenu"
						ref="slashMenuEl"
						class="pipeline-insights__slash-menu"
						role="listbox"
					>
						<div class="pipeline-insights__slash-menu-label">Charts</div>
						<button
							v-for="(command, index) in filteredSlashCommands"
							:key="command.id"
							type="button"
							class="pipeline-insights__slash-item"
							:class="{ 'pipeline-insights__slash-item--active': index === slashMenuIndex }"
							role="option"
							:aria-selected="index === slashMenuIndex"
							@click="insertSlashCommand(command)"
						>
							<BarChart3 class="pipeline-insights__slash-item-icon" />
							<span class="pipeline-insights__slash-item-body">
								<span class="pipeline-insights__slash-item-title">/{{ command.slash }}</span>
								<span class="pipeline-insights__slash-item-desc">{{ command.description }}</span>
							</span>
						</button>
						<div v-if="filteredSlashCommands.length === 0" class="pipeline-insights__slash-empty">
							No charts match “/{{ slashState?.query }}”
						</div>
					</div>
					<input
						ref="composerInput"
						v-model="draft"
						type="text"
						:placeholder="composerPlaceholder"
						class="pipeline-insights__input"
						@input="onComposerInput"
						@click="syncComposerCursor"
						@keyup="syncComposerCursor"
						@keydown="onComposerKeydown"
					/>
				</div>
				<button type="submit" class="pipeline-insights__send" :disabled="loading || !draft.trim()">
					<SendHorizontal class="w-4 h-4" />
				</button>
			</form>
		</div>
	</section>

	<Teleport to="body">
		<button
			v-if="!chatComposerVisible"
			type="button"
			class="pipeline-insights-jump"
			aria-label="Jump to chat input"
			@click="jumpToChat()"
		>
			<MessageSquareText class="pipeline-insights-jump__icon" />
			<span class="pipeline-insights-jump__label">Jump to chat</span>
		</button>
	</Teleport>
</template>
