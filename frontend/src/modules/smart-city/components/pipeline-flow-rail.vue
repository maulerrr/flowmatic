<script setup lang="ts">
import { inject } from 'vue'
import { Cpu, Database, Network, Radio } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-if="pipeline.selectedPipeline" class="pipeline-flow-rail pipeline-animate-rise">
		<button type="button" class="pipeline-node text-left" @click="pipeline.activeStage = 'sources'">
			<div class="pipeline-node__step">Stage 01</div>
			<div class="flex items-center gap-2 mt-2">
				<Radio class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Sources</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.sourceStatusText }} · {{ pipeline.sourceMixSummary }}</p>
		</button>
		<button type="button" class="pipeline-node text-left" @click="pipeline.activeStage = 'processing'">
			<div class="pipeline-node__step">Stage 02</div>
			<div class="flex items-center gap-2 mt-2">
				<Cpu class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Processing unit</div>
			</div>
			<p class="pipeline-node__meta truncate">{{ pipeline.activeModelLabel }}</p>
		</button>
		<button type="button" class="pipeline-node text-left" @click="pipeline.activeStage = 'lake'">
			<div class="pipeline-node__step">Stage 03</div>
			<div class="flex items-center gap-2 mt-2">
				<Database class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Lake & export</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.activeDataLake?.bucket ?? 'No output target' }}</p>
		</button>
		<button type="button" class="pipeline-node text-left" @click="pipeline.activeStage = 'federated'">
			<div class="pipeline-node__step">Stage 04</div>
			<div class="flex items-center gap-2 mt-2">
				<Network class="w-4 h-4 text-primary" />
				<div class="pipeline-node__title">Federated</div>
			</div>
			<p class="pipeline-node__meta">{{ pipeline.federatedConfig.status }} · {{ pipeline.federatedRounds.length }} rounds</p>
		</button>
	</section>
</template>
