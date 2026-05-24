<script setup lang="ts">
import { inject } from 'vue'
import { Cpu } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-show="pipeline.activeStage === 'processing'" class="pipeline-panel pipeline-panel__body space-y-5 flow-section flow-section--accent">
		<div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
			<div class="max-w-2xl">
				<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Step 2</p>
				<h3 class="mt-1 text-lg font-semibold text-foreground flex items-center gap-2">
					<Cpu class="w-4 h-4" /> Runtime Processing Unit
				</h3>
				<p class="mt-2 text-sm text-foreground/60">
					This stage preprocesses live events. It applies validation, cleaning, anomaly checks, and the active model before anything is exported.
				</p>
			</div>
			<div class="flex flex-wrap gap-2">
				<button @click="pipeline.showConfigModal = true" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground">
					Configure processing
				</button>
				<button
					@click="pipeline.testProcessingUnit"
					:disabled="!pipeline.selectedPipeline?.activeModelId"
					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Test with sample event
				</button>
			</div>
		</div>
		<div class="grid gap-5 xl:grid-cols-[0.85fr,1.15fr]">
			<div class="space-y-3">
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Runtime status</p>
					<div class="grid grid-cols-2 gap-3 mt-3 text-xs">
						<div class="rounded-lg border border-border bg-card p-3">Throughput<br /><b>{{ pipeline.throughput }} GB/s</b></div>
						<div class="rounded-lg border border-border bg-card p-3">Recent events<br /><b>{{ pipeline.events.length }}</b></div>
					</div>
					<div class="rounded-lg border border-border bg-card p-3 mt-3">
						<div class="text-[10px] uppercase tracking-wider text-foreground/40">Current processing model</div>
						<div class="mt-1 break-words text-xs font-mono text-foreground/70">{{ pipeline.activeModelLabel }}</div>
					</div>
					<div class="grid grid-cols-3 gap-2 text-[10px] font-bold uppercase tracking-wider mt-3">
						<span
							:class="[
								'rounded border px-2 py-1 text-center',
								pipeline.streamConfig.schemaValidation ? 'border-success/20 text-success' : 'border-border text-foreground/40',
							]"
							>Schema</span
						>
						<span
							:class="[
								'rounded border px-2 py-1 text-center',
								pipeline.streamConfig.autoCleaning ? 'border-success/20 text-success' : 'border-border text-foreground/40',
							]"
							>Clean</span
						>
						<span
							:class="[
								'rounded border px-2 py-1 text-center',
								pipeline.streamConfig.anomalyDetection ? 'border-success/20 text-success' : 'border-border text-foreground/40',
							]"
							>Detect</span
						>
					</div>
				</div>
				<div v-if="pipeline.latestProcessingResult" class="rounded-lg border border-white/5 bg-black/20 p-4 text-xs text-foreground/70 max-h-64 overflow-auto">
					<div class="text-[10px] uppercase tracking-wider text-foreground/40 mb-2">Latest processing output</div>
					<pre>{{ JSON.stringify(pipeline.latestProcessingResult.output?.result ?? pipeline.latestProcessingResult, null, 2) }}</pre>
				</div>
			</div>
			<div class="space-y-3">
				<div>
					<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Choose model used by the processing unit</p>
					<p class="mt-1 text-sm text-foreground/60">Only one model is active for live preprocessing at a time.</p>
				</div>
				<div v-if="pipeline.researchModels.length > 0" class="space-y-2 max-h-80 overflow-auto pr-1">
					<div
						v-for="model in pipeline.researchModels.slice(0, 6)"
						:key="model.id"
						:class="[
							'rounded-lg border p-3 text-xs',
							pipeline.selectedPipeline?.activeModelId === model.id ? 'border-primary/60 bg-primary/5' : 'border-border bg-surface-2',
						]"
					>
						<div class="flex items-center justify-between gap-2">
							<div class="font-semibold text-foreground truncate">{{ model.run }}</div>
							<span class="text-foreground/40">{{ model.hasTorchScript ? 'TorchScript' : 'PyTorch' }}</span>
						</div>
						<div class="text-foreground/50 font-mono mt-1">{{ model.kind }} / {{ model.dataset }}</div>
						<div class="flex items-center justify-between mt-2">
							<span v-if="pipeline.selectedPipeline?.activeModelId === model.id" class="rounded bg-primary/10 px-2 py-0.5 text-[10px] font-bold uppercase text-primary">
								Active
							</span>
							<span v-else class="text-foreground/40">Available</span>
							<button @click="pipeline.deployResearchModel(model)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">
								Use for processing
							</button>
						</div>
					</div>
				</div>
				<div v-if="pipeline.models.length > 0" class="space-y-2">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Local trained artifacts</div>
					<div
						v-for="model in pipeline.models.slice(0, 3)"
						:key="model.id"
						:class="[
							'rounded-lg border p-3 text-xs',
							pipeline.selectedPipeline?.activeModelId === model.id ? 'border-primary/60 bg-primary/5' : 'border-border bg-surface-2',
						]"
					>
						<div class="flex items-center justify-between gap-2">
							<div class="font-semibold text-foreground">{{ model.name }}</div>
							<div class="text-foreground/50 font-mono">{{ model.status }} / {{ model.version }}</div>
						</div>
						<div class="flex gap-2 mt-2">
							<button @click="pipeline.deployModel(model)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">
								Use for processing
							</button>
							<button @click="pipeline.promoteModel(model)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">
								Promote
							</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	</section>
</template>
