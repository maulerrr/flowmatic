<script setup lang="ts">
import { provide, reactive } from 'vue'

import '@/modules/smart-city/styles/pipeline-workbench.css'
import PipelineEmptyState from '@/modules/smart-city/components/pipeline-empty-state.vue'
import PipelineFederatedStage from '@/modules/smart-city/components/pipeline-federated-stage.vue'
import PipelineFlowRail from '@/modules/smart-city/components/pipeline-flow-rail.vue'
import PipelineLakeExportStage from '@/modules/smart-city/components/pipeline-lake-export-stage.vue'
import PipelineModals from '@/modules/smart-city/components/pipeline-modals.vue'
import PipelineObservabilityPanel from '@/modules/smart-city/components/pipeline-observability-panel.vue'
import PipelineProcessingStage from '@/modules/smart-city/components/pipeline-processing-stage.vue'
import PipelineSourcesStage from '@/modules/smart-city/components/pipeline-sources-stage.vue'
import PipelineSystemLog from '@/modules/smart-city/components/pipeline-system-log.vue'
import PipelineViewToolbar from '@/modules/smart-city/components/pipeline-view-toolbar.vue'
import PipelineWorkbenchHeader from '@/modules/smart-city/components/pipeline-workbench-header.vue'
import PipelineWorkflowView from '@/modules/smart-city/components/pipeline-workflow-view.vue'

import { useSmartCityPipeline } from '@/modules/smart-city/composables/useSmartCityPipeline'
import { PipelineWorkbenchKey } from '@/modules/smart-city/pipeline-context'

const pipeline = reactive(useSmartCityPipeline())
provide(PipelineWorkbenchKey, pipeline)
</script>

<template>
	<div class="pipeline-workbench space-y-6">
		<PipelineWorkbenchHeader />
		<PipelineViewToolbar />
		<PipelineFlowRail />

		<PipelineEmptyState v-if="!pipeline.selectedPipelineId && !pipeline.loading" />
		<PipelineWorkflowView v-else-if="pipeline.viewMode === 'workflow'" />
		<section v-else class="space-y-6 pipeline-animate-rise">
			<div class="pipeline-stage-tabs">
				<button
					type="button"
					:class="['pipeline-stage-tab', pipeline.activeStage === 'sources' ? 'pipeline-stage-tab--active' : '']"
					@click="pipeline.activeStage = 'sources'"
				>
					Sources
				</button>
				<button
					type="button"
					:class="['pipeline-stage-tab', pipeline.activeStage === 'processing' ? 'pipeline-stage-tab--active' : '']"
					@click="pipeline.activeStage = 'processing'"
				>
					Processing
				</button>
				<button
					type="button"
					:class="['pipeline-stage-tab', pipeline.activeStage === 'lake' ? 'pipeline-stage-tab--active' : '']"
					@click="pipeline.activeStage = 'lake'"
				>
					Lake & export
				</button>
				<button
					type="button"
					:class="['pipeline-stage-tab', pipeline.activeStage === 'federated' ? 'pipeline-stage-tab--active' : '']"
					@click="pipeline.activeStage = 'federated'"
				>
					Federated
				</button>
			</div>

			<div class="pipeline-panel">
				<div class="pipeline-panel__header">
					<p class="text-sm text-foreground/70">
						Operate one continuous pipeline: ingest through the sensor simulator or external feeds, preprocess in the runtime unit, persist medallion tiers to the lake,
						export through adapters, and coordinate federated rounds.
					</p>
				</div>
			</div>

			<PipelineObservabilityPanel />
			<PipelineSourcesStage />
			<PipelineProcessingStage />
			<PipelineLakeExportStage />
			<PipelineFederatedStage />
		</section>

		<PipelineSystemLog />
		<PipelineModals />
	</div>
</template>

<style scoped>
:deep(.flow-section--accent) {
	border-color: rgba(52, 208, 195, 0.18);
}
</style>
