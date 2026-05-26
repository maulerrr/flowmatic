<script setup lang="ts">
import { inject } from 'vue'
import { Cpu, Database, Network, Radio } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'
import PipelineLiveConduit from './pipeline-live-conduit.vue'
const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<div
		class="pipeline-flow-rail-wrap"
		:class="{ 'pipeline-flow-rail-wrap--live': pipeline.isPipelineLive }"
	>
		<PipelineLiveConduit v-if="pipeline.selectedPipeline" />
		<section v-if="pipeline.selectedPipeline" class="pipeline-flow-rail pipeline-animate-rise">
			<button
				type="button"
				:class="[
					'pipeline-node text-left',
					pipeline.activeStage === 'sources' ? 'pipeline-node--active' : '',
					pipeline.isPipelineLive ? 'pipeline-node--streaming' : '',
				]"
				@click="pipeline.activeStage = 'sources'"
			>
				<div class="pipeline-node__live-indicator" v-if="pipeline.isPipelineLive && pipeline.runningSourceCount > 0" />
				<div class="pipeline-node__step">01 · Ingest</div>
			<div class="flex items-center gap-2 mt-2">
				<Radio class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Sources</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.sourceStatusText }} · {{ pipeline.sourceMixSummary }}</p>
		</button>
		<button
			type="button"
			:class="[
				'pipeline-node text-left',
				pipeline.activeStage === 'processing' ? 'pipeline-node--active' : '',
				pipeline.isPipelineLive && pipeline.selectedPipeline?.activeModelId ? 'pipeline-node--streaming' : '',
			]"
			@click="pipeline.activeStage = 'processing'"
		>
			<div
				class="pipeline-node__live-indicator"
				v-if="pipeline.isPipelineLive && pipeline.selectedPipeline?.activeModelId"
			/>
			<div class="pipeline-node__step">02 · Process</div>
			<div class="flex items-center gap-2 mt-2">
				<Cpu class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Core unit</div>
			</div>
			<p class="pipeline-node__meta truncate">{{ pipeline.activeModelLabel }}</p>
			<span
				role="button"
				tabindex="0"
				class="pipeline-node__action"
				@click.stop="pipeline.openConfigModal()"
				@keydown.enter.stop="pipeline.openConfigModal()"
			>
				Configure
			</span>
		</button>
		<button
			type="button"
			:class="[
				'pipeline-node text-left',
				pipeline.activeStage === 'lake' ? 'pipeline-node--active' : '',
				pipeline.isPipelineLive && pipeline.continuousExportTargets.length > 0 ? 'pipeline-node--streaming' : '',
			]"
			@click="pipeline.activeStage = 'lake'"
		>
			<div
				class="pipeline-node__live-indicator"
				v-if="pipeline.isPipelineLive && pipeline.continuousExportTargets.length > 0"
			/>
			<div class="pipeline-node__step">03 · Store & export</div>
			<div class="flex items-center gap-2 mt-2">
				<Database class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Lake & export</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.activeDataLake?.bucket ?? 'No output target' }}</p>
		</button>
		<button
			type="button"
			:class="['pipeline-node text-left', pipeline.activeStage === 'federated' ? 'pipeline-node--active' : '']"
			@click="pipeline.activeStage = 'federated'"
		>
			<div class="pipeline-node__step">04 · Federated</div>
			<div class="flex items-center gap-2 mt-2">
				<Network class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Federated</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.federatedConfig.status }} · {{ pipeline.federatedRounds.length }} rounds</p>
		</button>
		</section>
	</div>
</template>
