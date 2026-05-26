<script setup lang="ts">
import { inject } from 'vue'
import { LayoutDashboard, ScrollText, Workflow } from 'lucide-vue-next'
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
		<p v-if="pipeline.viewMode === 'dashboard'" class="hidden lg:block text-xs text-foreground/45 max-w-md ml-3">
			Pick a stage below the flow map. Panels show only the active stage so the workbench stays focused.
		</p>
		<div class="flex flex-wrap gap-2 ml-auto">
			<button
				type="button"
				@click="pipeline.showObservability = !pipeline.showObservability"
				:class="[
					'pipeline-stage-tab',
					pipeline.showObservability ? 'pipeline-stage-tab--active' : '',
				]"
			>
				Metrics
			</button>
			<button
				type="button"
				@click="pipeline.showSystemLog = !pipeline.showSystemLog"
				:class="[
					'pipeline-stage-tab flex items-center gap-1.5',
					pipeline.showSystemLog ? 'pipeline-stage-tab--active' : '',
				]"
			>
				<ScrollText class="w-4 h-4" /> Log
			</button>
		</div>
	</div>
</template>
