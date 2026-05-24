<script setup lang="ts">
import { inject } from 'vue'
import { LayoutDashboard, Workflow } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<div class="pipeline-toolbar pipeline-animate-rise">
		<div class="flex items-center bg-card border border-border rounded-xl p-1">
			<button
				@click="pipeline.viewMode = 'dashboard'"
				:class="['pipeline-stage-tab', pipeline.viewMode === 'dashboard' ? 'pipeline-stage-tab--active' : '']"
			>
				<LayoutDashboard class="w-4 h-4 inline mr-1.5" /> Workbench
			</button>
			<button
				@click="pipeline.viewMode = 'workflow'"
				:class="['pipeline-stage-tab', pipeline.viewMode === 'workflow' ? 'pipeline-stage-tab--active' : '']"
			>
				<Workflow class="w-4 h-4 inline mr-1.5" /> Workflow graph
			</button>
		</div>
		<div class="hidden md:flex flex-wrap gap-2">
			<span class="pipeline-chip">Events: {{ pipeline.events.length }}</span>
			<span class="pipeline-chip">Models: {{ pipeline.processingModelCount }}</span>
			<span class="pipeline-chip">Exports: {{ pipeline.exportTargets.length }}</span>
		</div>
	</div>
</template>
