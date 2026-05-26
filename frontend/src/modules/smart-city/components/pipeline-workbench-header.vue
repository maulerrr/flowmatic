<script setup lang="ts">

import { inject } from 'vue'

import { useRouter } from 'vue-router'

import { Activity, MessageSquareText, Plus, RefreshCw, Trash2 } from 'lucide-vue-next'

import { PipelineWorkbenchKey } from '../pipeline-context'



const pipeline = inject(PipelineWorkbenchKey)

if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')



const router = useRouter()



function openInsights() {
	const pipelineId = pipeline?.selectedPipelineId
	if (!pipelineId) return
	void router.push({ path: '/insights', query: { pipelineId } })
}

</script>



<template>

	<header class="pipeline-hero pipeline-animate-rise">

		<button

			v-if="pipeline.selectedPipeline"

			type="button"

			class="pipeline-insights-launcher"

			title="Open insights chat for this pipeline"

			@click="openInsights"

		>

			<span class="pipeline-insights-launcher__dot" />

			<MessageSquareText class="w-3.5 h-3.5" />

			Ask insights

		</button>



		<div class="flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">

			<div class="space-y-3 max-w-2xl">

				<div class="flex flex-wrap items-center gap-2">

					<div class="pipeline-kicker">

						<Activity class="w-4 h-4" />

						<span>Pipeline workbench</span>

					</div>

					<span

						v-if="pipeline.isPipelineLive"

						class="pipeline-runtime-beacon"

						title="Pipeline is actively streaming events"

					>

						<span class="pipeline-runtime-beacon__ring" />

						<span class="pipeline-runtime-beacon__core" />

						Live

					</span>

					<span

						:class="[

							'pipeline-chip',

							pipeline.wsStatus === 'connected' ? 'pipeline-chip--live' : 'pipeline-chip--warn',

						]"

					>

						WS {{ pipeline.wsStatus }}

					</span>

					<span

						:class="[

							'pipeline-chip',

							pipeline.isPipelineLive ? 'pipeline-chip--live' : '',

						]"

					>

						{{ pipeline.pipelineRuntimeLabel }}

					</span>

				</div>

				<h1 class="pipeline-title text-foreground">{{ pipeline.selectedPipeline?.name ?? 'Smart city pipeline' }}</h1>

				<p class="text-sm text-foreground/60">

					{{ pipeline.pipelineFlowHint }}

				</p>

			</div>

			<div class="pipeline-toolbar w-full xl:w-auto">

				<select

					v-model="pipeline.selectedPipelineId"

					class="bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground min-w-56"

				>

					<option value="">No pipeline selected</option>

					<option v-for="p in pipeline.pipelines" :key="p.id" :value="p.id">{{ p.name }}</option>

				</select>

				<button

					@click="pipeline.showPipelineModal = true"

					class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium flex items-center gap-2"

				>

					<Plus class="w-4 h-4" /> Pipeline

				</button>

				<button

					v-if="pipeline.selectedPipeline"

					@click="pipeline.archivePipeline"

					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-red-400 flex items-center gap-2"

				>

					<Trash2 class="w-4 h-4" /> Archive

				</button>

				<button

					@click="

						pipeline.selectedPipelineId ? pipeline.loadDashboard(pipeline.selectedPipelineId) : pipeline.loadPipelines()

					"

					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground flex items-center gap-2"

				>

					<RefreshCw class="w-4 h-4" /> Refresh

				</button>

			</div>

		</div>

	</header>

</template>

