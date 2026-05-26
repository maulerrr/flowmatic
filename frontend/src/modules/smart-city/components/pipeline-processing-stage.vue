<script setup lang="ts">
import { inject } from 'vue'
import { Cpu, Loader2, Settings2 } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-show="pipeline.activeStage === 'processing'" class="pipeline-panel pipeline-panel__body space-y-5 flow-section flow-section--accent">
		<div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
			<div>
				<h3 class="text-lg font-semibold text-foreground flex items-center gap-2">
					<Cpu class="w-4 h-4" /> Core processing unit
				</h3>
				<p class="mt-1 text-sm text-foreground/60">Validates, cleans, and scores live events before export.</p>
			</div>
			<div class="flex flex-wrap gap-2">
				<button
					@click="pipeline.openConfigModal()"
					class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium flex items-center gap-2"
				>
					<Settings2 class="w-4 h-4" /> Configure core unit
				</button>
				<button
					@click="pipeline.testProcessingUnit"
					:disabled="pipeline.streamConfig.coreUnitMode !== 'auto' && !pipeline.selectedPipeline?.activeModelId"
					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Test sample event
				</button>
			</div>
		</div>

		<div class="grid gap-4 md:grid-cols-3">
			<div class="rounded-lg border border-border bg-surface-2 p-4 md:col-span-3">
				<div class="flex flex-col gap-4">
					<div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
						<div>
							<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Core unit mode</div>
							<p class="mt-1 text-xs text-foreground/55">
								Auto routes each sensor stream to a modality-safe trained model. Manual uses one Hugging Face checkpoint for all events.
							</p>
						</div>
						<div class="flex flex-col gap-2 sm:w-56">
							<div class="grid grid-cols-2 gap-2">
								<button
									type="button"
									:disabled="pipeline.saving"
									:class="[
										'rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50',
										pipeline.streamConfig.coreUnitMode !== 'auto'
											? 'border-primary/50 bg-primary/10 text-primary'
											: 'border-border text-foreground/60 hover:text-foreground',
									]"
									@click="pipeline.streamConfig.coreUnitMode = 'manual'"
								>
									Manual
								</button>
								<button
									type="button"
									:disabled="pipeline.saving"
									:class="[
										'rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50',
										pipeline.streamConfig.coreUnitMode === 'auto'
											? 'border-primary/50 bg-primary/10 text-primary'
											: 'border-border text-foreground/60 hover:text-foreground',
									]"
									@click="pipeline.streamConfig.coreUnitMode = 'auto'"
								>
									Auto
								</button>
							</div>
							<button
								type="button"
								@click="pipeline.saveConfiguration()"
								:disabled="pipeline.saving"
								class="rounded-lg border border-border px-3 py-2 text-xs font-medium text-foreground/70 hover:text-foreground disabled:opacity-40 flex items-center justify-center gap-2"
							>
								<Loader2 v-if="pipeline.isApplyingAutoRouting" class="w-3.5 h-3.5 animate-spin text-primary" />
								<span>{{
									pipeline.isApplyingAutoRouting
										? pipeline.autoRoutingPhase === 'matching'
											? 'Choosing models…'
											: 'Applying…'
										: 'Apply mode'
								}}</span>
							</button>
						</div>
					</div>

					<div
						v-if="pipeline.isApplyingAutoRouting && pipeline.streamConfig.coreUnitMode === 'auto'"
						class="flex items-start gap-3 rounded-lg border border-primary/25 bg-primary/10 px-4 py-3 text-sm text-primary"
					>
						<Loader2 class="mt-0.5 w-4 h-4 shrink-0 animate-spin" />
						<div>
							<div class="font-semibold">Choosing the best models</div>
							<p class="mt-1 text-xs text-primary/80">{{ pipeline.autoRoutingStatusMessage }}</p>
						</div>
					</div>

					<div
						v-else-if="pipeline.streamConfig.coreUnitMode === 'auto' && pipeline.autoRoutingBindings.length"
						class="space-y-3 rounded-lg border border-primary/20 bg-primary/5 px-4 py-3"
					>
						<div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
							<div class="text-sm font-semibold text-foreground">Configured routes</div>
							<span class="text-[10px] font-bold uppercase tracking-wider text-success">Ready</span>
						</div>
						<p v-if="pipeline.autoRoutingSummary" class="text-xs text-foreground/60">{{ pipeline.autoRoutingSummary }}</p>
						<ul class="grid gap-2 sm:grid-cols-2">
							<li
								v-for="binding in pipeline.autoRoutingBindings"
								:key="`${binding.sensorKind}-${binding.modelId}`"
								class="rounded-md border border-border/70 bg-surface-2 px-3 py-2 text-xs"
							>
								<div class="font-semibold text-foreground">{{ binding.sensorKind }} → {{ binding.label }}</div>
								<div v-if="binding.reason" class="mt-1 text-foreground/50">{{ binding.reason }}</div>
							</li>
						</ul>
						<p v-if="pipeline.lastAutoResolution?.label" class="text-xs text-foreground/50">
							Last live routing: {{ pipeline.lastAutoResolution.sensorKind }} → {{ pipeline.lastAutoResolution.label }}
						</p>
					</div>

					<p
						v-else-if="pipeline.streamConfig.coreUnitMode === 'auto'"
						class="text-xs text-foreground/50 rounded-lg border border-dashed border-border px-4 py-3"
					>
						Click <span class="font-medium text-foreground/70">Apply mode</span> to match your Stage 01 sensor sources with trained checkpoints.
					</p>
				</div>
			</div>

			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">
					{{ pipeline.streamConfig.coreUnitMode === 'auto' ? 'Active routing' : 'Active model' }}
				</div>
				<div class="mt-2 text-sm font-semibold text-foreground break-words">
					<template v-if="pipeline.isApplyingAutoRouting && pipeline.streamConfig.coreUnitMode === 'auto'">
						<span class="inline-flex items-center gap-2 text-primary">
							<Loader2 class="w-3.5 h-3.5 animate-spin" />
							Selecting best models…
						</span>
					</template>
					<template v-else>
						{{ pipeline.activeModelLabel }}
					</template>
				</div>
				<p v-if="pipeline.activeModelSubLabel" class="mt-2 text-xs leading-relaxed text-foreground/55">
					{{ pipeline.activeModelSubLabel }}
				</p>
				<div class="mt-2 text-[10px] uppercase tracking-wider text-foreground/45">
					Mode: {{ pipeline.streamConfig.coreUnitMode === 'auto' ? 'Adaptive auto' : 'Manual' }}
				</div>
				<button @click="pipeline.openConfigModal()" class="mt-3 text-xs text-primary hover:underline">
					{{ pipeline.streamConfig.coreUnitMode === 'auto' ? 'Review routes' : 'Change model' }}
				</button>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Throughput</div>
				<div class="mt-2 text-2xl font-semibold text-foreground">{{ pipeline.throughput }} <span class="text-sm text-foreground/50">GB/s</span></div>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Guards</div>
				<div class="mt-2 flex flex-wrap gap-2 text-[10px] font-bold uppercase tracking-wider">
					<span
						:class="[
							'rounded border px-2 py-1',
							pipeline.streamConfig.schemaValidation ? 'border-success/20 text-success' : 'border-border text-foreground/40',
						]"
						>Schema</span
					>
					<span
						:class="[
							'rounded border px-2 py-1',
							pipeline.streamConfig.autoCleaning ? 'border-success/20 text-success' : 'border-border text-foreground/40',
						]"
						>Clean</span
					>
					<span
						:class="[
							'rounded border px-2 py-1',
							pipeline.streamConfig.anomalyDetection ? 'border-success/20 text-success' : 'border-border text-foreground/40',
						]"
						>Detect</span
					>
				</div>
			</div>
		</div>

		<div v-if="pipeline.processingError" class="rounded-lg border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-300">
			<div class="text-[10px] font-bold uppercase tracking-wider text-red-400/80 mb-2">Inference issue</div>
			<p class="text-xs leading-relaxed">{{ pipeline.processingError }}</p>
		</div>

		<div v-if="pipeline.latestProcessingResult && !pipeline.latestProcessingResult.error" class="rounded-lg border border-white/5 bg-black/20 p-4 text-xs text-foreground/70 max-h-48 overflow-auto">
			<div class="text-[10px] uppercase tracking-wider text-foreground/40 mb-2">Latest output</div>
			<pre>{{ JSON.stringify(pipeline.latestProcessingResult.output?.result ?? pipeline.latestProcessingResult.output ?? pipeline.latestProcessingResult, null, 2) }}</pre>
		</div>
	</section>
</template>
