<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowUpRight, MessageSquareText, Network, Sparkles } from 'lucide-vue-next'
import { toast } from 'vue-sonner'

import { apiClient, type SmartCityPipeline } from '@/api/client'
import PipelineInsightConfig from '@/modules/smart-city/components/pipeline-insight-config.vue'
import PipelineInsightFeed from '@/modules/smart-city/components/pipeline-insight-feed.vue'
import PipelineInsightsChat from '@/modules/smart-city/components/pipeline-insights-chat.vue'
import { usePipelineInsights } from '@/modules/smart-city/composables/usePipelineInsights'
import '@/modules/smart-city/styles/pipeline-insights.css'

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const pipelines = ref<SmartCityPipeline[]>([])
const selectedPipelineId = ref('')

const insights = usePipelineInsights(() => selectedPipelineId.value)
const {
	config: insightConfig,
	loading: insightsLoading,
	running: insightsRunning,
	runs: insightRuns,
	selectedRunId: insightSelectedRunId,
	saveConfig: saveInsightConfig,
	runNow: runInsightNow,
} = insights

const selectedPipeline = computed(() =>
	pipelines.value.find(pipeline => pipeline.id === selectedPipelineId.value),
)

function syncRouteQuery(pipelineId: string) {
	const current = typeof route.query.pipelineId === 'string' ? route.query.pipelineId : ''
	if (current === pipelineId) return
	void router.replace({
		path: '/insights',
		query: pipelineId ? { pipelineId } : {},
	})
}

async function loadPipelines() {
	loading.value = true
	try {
		const response = await apiClient.listSmartCityPipelines()
		pipelines.value = response.data ?? []
		const queryId = typeof route.query.pipelineId === 'string' ? route.query.pipelineId : ''
		const preferred =
			(queryId && pipelines.value.some(pipeline => pipeline.id === queryId) && queryId) ||
			pipelines.value[0]?.id ||
			''
		selectedPipelineId.value = preferred
		if (preferred) syncRouteQuery(preferred)
	} catch (error) {
		toast.error(error instanceof Error ? error.message : 'Could not load pipelines')
	} finally {
		loading.value = false
	}
}

function openWorkbench() {
	if (!selectedPipelineId.value) return
	void router.push({ path: '/connectors', query: { pipelineId: selectedPipelineId.value } })
}

watch(selectedPipelineId, pipelineId => {
	if (pipelineId) syncRouteQuery(pipelineId)
})

watch(
	() => route.query.pipelineId,
	pipelineId => {
		if (typeof pipelineId !== 'string' || !pipelineId || pipelineId === selectedPipelineId.value) return
		if (pipelines.value.some(pipeline => pipeline.id === pipelineId)) {
			selectedPipelineId.value = pipelineId
		}
	},
)

onMounted(() => {
	void loadPipelines()
})
</script>

<template>
	<div class="insights-studio">
		<header class="insights-studio__masthead">
			<div class="pl-4">
				<div class="insights-studio__kicker">
					<Sparkles class="w-3.5 h-3.5" />
					Observatory
				</div>
				<h1 class="insights-studio__title">Pipeline Insights</h1>
				<p class="insights-studio__lede">
					Scheduled AI analysis profiles your data, discovers patterns, engineers features, and renders adaptive charts — plus interactive chat.
				</p>
			</div>

			<div class="insights-studio__controls">
				<label class="text-[0.68rem] uppercase tracking-[0.12em] text-foreground/45 font-semibold">
					Pipeline channel
				</label>
				<select v-model="selectedPipelineId" class="insights-studio__select" :disabled="loading || pipelines.length === 0">
					<option value="" disabled>Select a pipeline</option>
					<option v-for="pipeline in pipelines" :key="pipeline.id" :value="pipeline.id">
						{{ pipeline.name }}
					</option>
				</select>
				<button
					type="button"
					class="insights-studio__workbench-link"
					:disabled="!selectedPipelineId"
					@click="openWorkbench"
				>
					<Network class="w-4 h-4" />
					Open in workbench
					<ArrowUpRight class="w-3.5 h-3.5 opacity-70" />
				</button>
			</div>
		</header>

		<div v-if="loading" class="insights-studio__empty">Loading pipeline channels…</div>
		<div v-else-if="pipelines.length === 0" class="insights-studio__empty">
			<MessageSquareText class="w-8 h-8 mx-auto mb-3 opacity-40" />
			<p>No pipelines yet. Create one in the workbench to start an insights session.</p>
		</div>
		<div v-else-if="selectedPipeline" class="insights-studio__stack">
			<PipelineInsightConfig
				:config="insightConfig"
				:loading="insightsLoading"
				:running="insightsRunning"
				@save="saveInsightConfig"
				@run-now="runInsightNow"
			/>
			<PipelineInsightFeed
				:runs="insightRuns"
				:selected-run-id="insightSelectedRunId"
				:loading="insightsLoading"
				@select-run="insightSelectedRunId = $event"
			/>
			<PipelineInsightsChat
				:pipeline-id="selectedPipeline.id"
				:pipeline-name="selectedPipeline.name"
			/>
		</div>
	</div>
</template>
