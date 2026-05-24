<script setup lang="ts">
import { inject } from 'vue'
import { Activity, Plus, RefreshCw, Trash2 } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<header class="pipeline-hero pipeline-animate-rise">
		<div class="flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
			<div class="space-y-4 max-w-3xl">
				<div class="flex flex-wrap items-center gap-2">
					<div class="pipeline-kicker">
						<Activity class="w-4 h-4" />
						<span>Pipeline workbench</span>
					</div>
					<span class="pipeline-chip pipeline-chip--live">Live ingest</span>
					<span class="pipeline-chip">Source → Core → Export</span>
				</div>
				<h1 class="pipeline-title text-foreground">Smart City Pipeline Control</h1>
				<p class="text-sm md:text-base text-foreground/65 leading-relaxed">
					Operate the full medallion flow in one surface: simulated or external sensor feeds, runtime processing, data lake
					storage, adapter exports, and federated training rounds.
				</p>
				<div class="flex flex-wrap gap-2">
					<span class="pipeline-chip">Pipeline: {{ pipeline.selectedPipeline?.name ?? 'None selected' }}</span>
					<span class="pipeline-chip">{{ pipeline.sourceStatusText }}</span>
					<span class="pipeline-chip">{{ pipeline.throughput }} GB/s</span>
					<span class="pipeline-chip">Lake: {{ pipeline.activeDataLake?.name ?? 'Not connected' }}</span>
					<span
						:class="[
							'pipeline-chip',
							pipeline.wsStatus === 'connected' ? 'pipeline-chip--live' : 'pipeline-chip--warn',
						]"
					>
						WS {{ pipeline.wsStatus }}
					</span>
				</div>
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
