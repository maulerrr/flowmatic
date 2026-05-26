<script setup lang="ts">
import { computed, inject } from 'vue'
import { ArrowRight } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')

const guide = computed(() => {
	switch (pipeline.activeStage) {
		case 'sources':
			return {
				title: 'Ingest sensor feeds',
				body: 'Add sources and start them. Events flow into the core unit when the pipeline is running.',
				cta: 'Add source',
				action: () => {
					pipeline.showSourceModal = true
				},
			}
		case 'processing':
			return {
				title: 'Run inference on each event',
				body: 'Deploy a model to the core unit. Cleaned and business rows are produced automatically.',
				cta: 'Configure core unit',
				action: () => pipeline.openConfigModal(),
			}
		case 'lake':
			return {
				title: 'Persist and export outcomes',
				body: 'Lake writes happen in realtime. Export targets push batches to HF, Postgres, or Mongo on a cadence.',
				cta: 'Manage exports',
				action: () => {
					pipeline.lakeExportTab = 'export'
					pipeline.openExportTargetModal()
				},
			}
		case 'federated':
			return {
				title: 'Coordinate federated rounds',
				body: 'Connect a coordinator, start rounds, and sync global model weights across nodes.',
				cta: 'Open federated workspace',
				action: () => pipeline.openFederatedWorkspace(),
			}
		default:
			return { title: '', body: '', cta: '', action: () => {} }
	}
})
</script>

<template>
	<section v-if="pipeline.selectedPipeline && pipeline.viewMode === 'dashboard'" class="pipeline-stage-guide pipeline-animate-rise">
		<div class="pipeline-stage-guide__badge">Stage focus</div>
		<div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
			<div class="max-w-2xl">
				<h2 class="pipeline-stage-guide__title">{{ guide.title }}</h2>
				<p class="pipeline-stage-guide__body">{{ guide.body }}</p>
			</div>
			<button type="button" class="pipeline-stage-guide__cta" @click="guide.action">
				{{ guide.cta }} <ArrowRight class="w-4 h-4" />
			</button>
		</div>
	</section>
</template>
