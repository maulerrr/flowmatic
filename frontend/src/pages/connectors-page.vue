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
import PipelineStageGuide from '@/modules/smart-city/components/pipeline-stage-guide.vue'
import PipelineStatusStrip from '@/modules/smart-city/components/pipeline-status-strip.vue'
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
	<div class="pipeline-workbench space-y-5" :class="{ 'pipeline-workbench--live': pipeline.isPipelineLive }">
		<PipelineWorkbenchHeader />
		<PipelineStatusStrip />
		<PipelineViewToolbar />
		<PipelineFlowRail />
		<PipelineStageGuide />

		<PipelineEmptyState v-if="!pipeline.selectedPipelineId && !pipeline.loading" />
		<PipelineWorkflowView v-else-if="pipeline.viewMode === 'workflow'" />
		<section v-else class="space-y-5 pipeline-animate-rise">
			<PipelineObservabilityPanel v-if="pipeline.showObservability" />
			<PipelineSourcesStage />
			<PipelineProcessingStage />
			<PipelineLakeExportStage />
			<PipelineFederatedStage />
		</section>

		<PipelineSystemLog v-if="pipeline.showSystemLog" />
		<PipelineModals />
	</div>
</template>

<style scoped>
:deep(.flow-section--accent) {
	border-color: rgba(52, 208, 195, 0.18);
}
</style>
