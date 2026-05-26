<script setup lang="ts">
import { computed, inject } from 'vue'
import { Activity, ArrowRight, Database, Radio, Zap } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')

const statusClass = computed(() => {
	switch (pipeline.pipelineRuntimeStatus) {
		case 'ACTIVE':
			return 'pipeline-status-strip--live'
		case 'PAUSED':
			return 'pipeline-status-strip--paused'
		case 'ERROR':
			return 'pipeline-status-strip--error'
		default:
			return 'pipeline-status-strip--idle'
	}
})
</script>

<template>
	<section v-if="pipeline.selectedPipeline" class="pipeline-status-strip pipeline-animate-rise" :class="statusClass">
		<div class="pipeline-status-strip__beacon">
			<span class="pipeline-status-strip__dot" />
			<div>
				<div class="pipeline-status-strip__label">{{ pipeline.pipelineRuntimeLabel }}</div>
				<div class="pipeline-status-strip__hint">{{ pipeline.pipelineFlowHint }}</div>
			</div>
		</div>

		<div class="pipeline-status-strip__metrics">
			<div class="pipeline-status-metric">
				<Radio class="w-3.5 h-3.5" />
				<span>{{ pipeline.sourceStatusText }}</span>
			</div>
			<div class="pipeline-status-metric">
				<Activity class="w-3.5 h-3.5" />
				<span>{{ pipeline.events.length }} events buffered</span>
			</div>
			<div class="pipeline-status-metric">
				<Database class="w-3.5 h-3.5" />
				<span>{{ pipeline.exportHealthSummary.continuous }} continuous exports</span>
			</div>
			<div class="pipeline-status-metric">
				<Zap class="w-3.5 h-3.5" />
				<span>WS {{ pipeline.wsStatus }}</span>
			</div>
		</div>

		<div class="pipeline-status-strip__actions">
			<button
				v-if="!pipeline.isPipelineLive"
				type="button"
				class="pipeline-status-strip__cta"
				:disabled="pipeline.saving || !pipeline.selectedPipelineId"
				@click="pipeline.startPipelineRuntime"
			>
				Start pipeline
			</button>
			<button
				v-else
				type="button"
				class="pipeline-status-strip__cta pipeline-status-strip__cta--ghost"
				:disabled="pipeline.saving"
				@click="pipeline.stopPipelineRuntime"
			>
				Stop
			</button>
			<button type="button" class="pipeline-status-strip__link" @click="pipeline.activeStage = 'lake'">
				Exports <ArrowRight class="w-3.5 h-3.5" />
			</button>
		</div>
	</section>
</template>
