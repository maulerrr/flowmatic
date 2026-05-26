<script setup lang="ts">
import { inject } from 'vue'
import { Cpu, Loader2, Save, X } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<Teleport to="body">
		<div v-if="pipeline.showPipelineModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showPipelineModal = false"></div>
		<div class="relative w-full max-w-md bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground">Create Pipeline</h2>
				<button @click="pipeline.showPipelineModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
			<input v-model="pipeline.pipelineForm.name" type="text" placeholder="Astana Traffic Intelligence" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
			<textarea v-model="pipeline.pipelineForm.description" rows="3" placeholder="What this pipeline processes" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground"></textarea>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showPipelineModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button @click="pipeline.createPipeline" :disabled="pipeline.saving" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium">Create</button>
			</div>
		</div>
	</div>

	<div v-if="pipeline.showSourceModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showSourceModal = false"></div>
		<div class="relative w-full max-w-xl bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground">Connect Source</h2>
				<button @click="pipeline.showSourceModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
			<input v-model="pipeline.sourceForm.name" type="text" placeholder="North Avenue Air Quality" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
			<div class="grid grid-cols-2 gap-2">
				<button
					@click="pipeline.sourceForm.type = 'WEBSOCKET'"
					:class="pipeline.sourceForm.type === 'WEBSOCKET' ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-2 border-border text-foreground/70'"
					class="px-3 py-2 rounded-lg border text-sm"
				>
					WebSocket
				</button>
				<button
					@click="pipeline.sourceForm.type = 'HTTP_POLLING'"
					:class="pipeline.sourceForm.type === 'HTTP_POLLING' ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-2 border-border text-foreground/70'"
					class="px-3 py-2 rounded-lg border text-sm"
				>
					HTTP Polling
				</button>
			</div>
			<select v-model="pipeline.sourceForm.sensorKind" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground">
				<option value="iot">Air Quality</option>
				<option value="video">Traffic Video</option>
				<option value="power">Power Grid</option>
				<option value="network">Network</option>
				<option value="weather">Weather</option>
				<option value="parking">Parking</option>
				<option value="traffic">Astana Traffic (Geo WS)</option>
			</select>
			<p v-if="pipeline.sourceForm.sensorKind === 'traffic'" class="text-xs text-foreground/55 leading-relaxed">
				Streams semi-synthetic Astana traffic events from the real-derived CSV dataset with latitude/longitude for AI map and heatmap analysis.
			</p>
			<div class="grid grid-cols-2 gap-2">
				<button
					@click="pipeline.sourceForm.mode = 'SIMULATED'"
					:class="pipeline.sourceForm.mode === 'SIMULATED' ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-2 border-border text-foreground/70'"
					class="px-3 py-2 rounded-lg border text-sm"
				>
					Simulated
				</button>
				<button
					@click="pipeline.sourceForm.mode = 'EXTERNAL'"
					:class="pipeline.sourceForm.mode === 'EXTERNAL' ? 'bg-primary/10 border-primary text-primary' : 'bg-surface-2 border-border text-foreground/70'"
					class="px-3 py-2 rounded-lg border text-sm"
				>
					External
				</button>
			</div>
			<input
				v-if="pipeline.sourceForm.mode === 'EXTERNAL'"
				v-model="pipeline.sourceForm.endpoint"
				:placeholder="pipeline.sourceForm.type === 'WEBSOCKET' ? 'wss://sensor.example/stream' : 'https://sensor.example/poll'"
				class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground"
			/>
			<div v-else class="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs text-foreground/70">
				<div class="font-semibold text-primary mb-1">Sensor simulator service</div>
				<p>Simulated sources route through the standalone demo service. Resolved endpoint:</p>
				<p class="mt-1 font-mono break-all text-foreground/80">{{ pipeline.simulatorEndpointHint }}</p>
			</div>
			<input v-model.number="pipeline.sourceForm.pollIntervalMs" type="number" min="1000" step="1000" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
			<div v-if="pipeline.sourceForm.mode === 'EXTERNAL'" class="grid gap-3 sm:grid-cols-2">
				<input v-model="pipeline.sourceForm.payloadPath" type="text" placeholder="Payload path, e.g. data.items" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.sourceForm.locationField" type="text" placeholder="Location field, e.g. metadata.city" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input
					v-if="pipeline.sourceForm.type === 'WEBSOCKET'"
					v-model="pipeline.sourceForm.subscribeMessage"
					type="text"
					placeholder='Optional subscribe message, e.g. {"type":"subscribe"}'
					class="sm:col-span-2 bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground"
				/>
				<select v-else v-model="pipeline.sourceForm.method" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground">
					<option value="GET">GET</option>
					<option value="POST">POST</option>
				</select>
			</div>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showSourceModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button @click="pipeline.addSource" :disabled="pipeline.saving" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium">Connect</button>
			</div>
		</div>
	</div>

	<div v-if="pipeline.showDataLakeModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showDataLakeModal = false"></div>
		<div class="relative w-full max-w-lg bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground">Connect S3 Data Lake</h2>
				<button @click="pipeline.showDataLakeModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
			<div class="grid grid-cols-2 gap-3">
				<input v-model="pipeline.dataLakeForm.name" placeholder="Lake name" class="col-span-2 bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<select v-model="pipeline.dataLakeForm.provider" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground">
					<option value="CUSTOM_S3">Custom S3</option>
					<option value="AWS_S3">AWS S3</option>
					<option value="MINIO">MinIO</option>
					<option value="R2">Cloudflare R2</option>
				</select>
				<input v-model="pipeline.dataLakeForm.bucket" placeholder="Bucket" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.dataLakeForm.region" placeholder="Region" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.dataLakeForm.endpoint" placeholder="Endpoint" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.dataLakeForm.basePrefix" placeholder="Base prefix" class="col-span-2 bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.dataLakeForm.accessKey" placeholder="Access key" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
				<input v-model="pipeline.dataLakeForm.secretKey" type="password" placeholder="Secret key" class="bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground" />
			</div>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showDataLakeModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button @click="pipeline.saveDataLake" :disabled="pipeline.saving" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium">Save</button>
			</div>
		</div>
	</div>

	<div v-if="pipeline.showConfigModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showConfigModal = false"></div>
		<div class="relative w-full max-w-lg bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5 max-h-[90vh] overflow-y-auto">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground flex items-center gap-2"><Cpu class="w-5 h-5 text-primary" /> Core Unit</h2>
				<button @click="pipeline.showConfigModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>

			<div class="space-y-2">
				<p class="text-xs font-bold uppercase tracking-wider text-foreground/40">Core unit mode</p>
				<div class="grid grid-cols-2 gap-2">
					<button
						type="button"
						:class="[
							'rounded-lg border px-3 py-2 text-sm font-medium transition-colors',
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
						:class="[
							'rounded-lg border px-3 py-2 text-sm font-medium transition-colors',
							pipeline.streamConfig.coreUnitMode === 'auto'
								? 'border-primary/50 bg-primary/10 text-primary'
								: 'border-border text-foreground/60 hover:text-foreground',
						]"
						@click="pipeline.streamConfig.coreUnitMode = 'auto'"
					>
						Auto
					</button>
				</div>
				<p class="text-xs text-foreground/50">
					<template v-if="pipeline.streamConfig.coreUnitMode === 'auto'">
						Adaptive routing sends traffic, weather, and geo streams to modality-safe trained checkpoints. Rules apply instantly; OpenAI can refine bindings when configured.
					</template>
					<template v-else>
						Pick a Hugging Face model manually. Every event uses the same checkpoint until you change it.
					</template>
				</p>
			</div>

			<div v-if="pipeline.streamConfig.coreUnitMode === 'auto'" class="space-y-3 rounded-lg border border-primary/20 bg-primary/5 p-4">
				<div class="text-sm font-semibold text-foreground">Adaptive routing preview</div>

				<div
					v-if="pipeline.isApplyingAutoRouting"
					class="flex items-start gap-3 rounded-md border border-primary/25 bg-primary/10 px-3 py-3 text-sm text-primary"
				>
					<Loader2 class="mt-0.5 w-4 h-4 shrink-0 animate-spin" />
					<div>
						<div class="font-semibold">Choosing the best models</div>
						<p class="mt-1 text-xs text-primary/80">{{ pipeline.autoRoutingStatusMessage }}</p>
					</div>
				</div>

				<template v-else>
					<p class="text-xs text-foreground/60">
						{{ pipeline.autoRoutingSummary || 'Apply to generate sensor-to-model bindings from your trained registry.' }}
					</p>
					<ul v-if="pipeline.autoRoutingBindings.length" class="space-y-2 text-xs">
						<li
							v-for="binding in pipeline.autoRoutingBindings"
							:key="`${binding.sensorKind}-${binding.modelId}`"
							class="rounded-md border border-border/70 bg-surface-2 px-3 py-2"
						>
							<div class="font-semibold text-foreground">{{ binding.sensorKind }} → {{ binding.label }}</div>
							<div class="text-foreground/50">{{ binding.reason }}</div>
						</li>
					</ul>
				</template>
			</div>

			<div v-else class="space-y-2">
				<label class="text-sm font-medium text-foreground">Processing model (Hugging Face)</label>
				<p class="text-xs text-foreground/50">
					Pull a model from your Hugging Face account. The core unit runs inference through the universal model-inference service.
				</p>

				<div
					v-if="!pipeline.huggingFaceIntegration.configured"
					class="rounded-lg border border-warning/30 bg-warning/5 px-3 py-3 text-xs text-foreground/70 space-y-2"
				>
					<p>Add your Hugging Face token in Settings (or during signup) to browse your models.</p>
					<input
						v-model="pipeline.hfTokenDraft"
						type="password"
						placeholder="hf_..."
						class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
					/>
					<button
						type="button"
						@click="pipeline.saveHuggingFaceToken(pipeline.hfTokenDraft)"
						class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium"
					>
						Save token
					</button>
				</div>

				<template v-else>
					<div class="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs text-foreground/70">
						Connected as <span class="font-semibold text-primary">@{{ pipeline.huggingFaceIntegration.username ?? 'hub-user' }}</span>
						<span v-if="pipeline.huggingFaceIntegration.tokenPreview"> · {{ pipeline.huggingFaceIntegration.tokenPreview }}</span>
					</div>

					<div class="flex gap-2">
						<input
							v-model="pipeline.hfModelSearch"
							type="search"
							placeholder="Search your Hub models"
							class="flex-1 bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
							@keyup.enter="pipeline.loadHuggingFaceCatalog(1)"
						/>
						<button
							type="button"
							@click="pipeline.loadHuggingFaceCatalog(1)"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground"
						>
							Search
						</button>
					</div>

					<div v-if="pipeline.hfModelLoading" class="text-sm text-foreground/50 py-4 text-center">Loading catalogue…</div>
					<div v-else-if="pipeline.hfModelCatalog?.items.length" class="space-y-2 max-h-56 overflow-auto pr-1">
						<button
							v-for="model in pipeline.hfModelCatalog.items"
							:key="model.id"
							type="button"
							@click="pipeline.selectHuggingFaceModel(model)"
							:class="[
								'w-full text-left rounded-lg border p-3 text-xs transition-colors',
								pipeline.hfSelectedModelId === model.modelId
									? 'border-primary/60 bg-primary/5'
									: 'border-border bg-surface-2 hover:border-primary/30',
							]"
						>
							<div class="flex items-start justify-between gap-2">
								<div class="font-semibold text-foreground break-all">{{ model.modelId }}</div>
								<span
									:class="[
										'shrink-0 rounded px-2 py-0.5 uppercase text-[10px] font-bold',
										model.resources.recommendedDevice === 'gpu'
											? 'bg-violet-500/10 text-violet-300'
											: 'bg-success/10 text-success',
									]"
								>
									{{ model.resources.recommendedDevice }}
								</span>
							</div>
							<div class="mt-1 text-foreground/50">
								{{ model.pipelineTag || model.library || 'general' }} · {{ model.downloads }} downloads
							</div>
							<div class="mt-1 text-foreground/40">
								RAM ~{{ model.resources.estimatedRamGb }} GB
								<span v-if="model.resources.estimatedVramGb"> · VRAM ~{{ model.resources.estimatedVramGb }} GB</span>
								· storage ~{{ model.resources.estimatedStorageGb }} GB
							</div>
						</button>
					</div>
					<div v-else class="rounded-lg border border-dashed border-border p-4 text-xs text-foreground/50">
						No models found. Try another search or enter a model ID manually below.
					</div>

					<div v-if="pipeline.hfModelCatalog && pipeline.hfModelCatalog.total > pipeline.hfModelCatalog.limit" class="flex items-center justify-between text-xs">
						<button
							type="button"
							:disabled="pipeline.hfModelPage <= 1"
							@click="pipeline.loadHuggingFaceCatalog(pipeline.hfModelPage - 1)"
							class="px-2 py-1 rounded border border-border disabled:opacity-40"
						>
							Previous
						</button>
						<span class="text-foreground/50">Page {{ pipeline.hfModelPage }}</span>
						<button
							type="button"
							:disabled="!pipeline.hfModelCatalog.hasMore"
							@click="pipeline.loadHuggingFaceCatalog(pipeline.hfModelPage + 1)"
							class="px-2 py-1 rounded border border-border disabled:opacity-40"
						>
							Next
						</button>
					</div>

					<div class="flex gap-2">
						<input
							v-model="pipeline.hfManualModelId"
							type="text"
							placeholder="username/model-name"
							class="flex-1 bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground font-mono"
						/>
						<button
							type="button"
							@click="pipeline.resolveManualHuggingFaceModel()"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground"
						>
							Use
						</button>
					</div>

					<div v-if="pipeline.hfSelectedModelDetails" class="rounded-lg border border-border bg-surface-2 p-3 text-xs space-y-1">
						<div class="font-semibold text-foreground">Selected: {{ pipeline.hfSelectedModelDetails.modelId }}</div>
						<div class="text-foreground/50">
							Provider: {{ pipeline.hfSelectedModelDetails.resources.inferenceProvider.replace('-', ' ') }}
						</div>
						<ul class="text-foreground/40 list-disc pl-4">
							<li v-for="(note, index) in pipeline.hfSelectedModelDetails.resources.notes" :key="index">{{ note }}</li>
						</ul>
					</div>
				</template>
			</div>

			<div class="space-y-3">
				<p class="text-xs font-bold uppercase tracking-wider text-foreground/40">Pipeline guards</p>
				<label class="flex items-center justify-between p-3 rounded-lg border border-border">
					<span class="font-medium text-foreground text-sm">Anomaly Detection</span>
					<input v-model="pipeline.streamConfig.anomalyDetection" type="checkbox" class="accent-primary" />
				</label>
				<label class="flex items-center justify-between p-3 rounded-lg border border-border">
					<span class="font-medium text-foreground text-sm">Schema Validation</span>
					<input v-model="pipeline.streamConfig.schemaValidation" type="checkbox" class="accent-primary" />
				</label>
				<label class="flex items-center justify-between p-3 rounded-lg border border-border">
					<span class="font-medium text-foreground text-sm">Auto Cleaning</span>
					<input v-model="pipeline.streamConfig.autoCleaning" type="checkbox" class="accent-primary" />
				</label>
			</div>

			<div>
				<div class="flex justify-between text-sm mb-2">
					<span class="text-foreground font-medium">Throughput Limit</span>
					<span class="font-mono text-primary">{{ pipeline.streamConfig.throughputLimit }} GB/s</span>
				</div>
				<input v-model.number="pipeline.streamConfig.throughputLimit" type="range" min="1" max="20" step="0.5" class="w-full accent-primary" />
			</div>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showConfigModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button @click="pipeline.saveConfiguration" :disabled="pipeline.saving" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium flex items-center gap-2 disabled:opacity-50">
					<Loader2 v-if="pipeline.isApplyingAutoRouting" class="w-4 h-4 animate-spin" />
					<Save v-else class="w-4 h-4" />
					{{
						pipeline.isApplyingAutoRouting
							? pipeline.autoRoutingPhase === 'matching'
								? 'Choosing models…'
								: 'Applying…'
							: 'Apply'
					}}
				</button>
			</div>
		</div>
	</div>

	<div v-if="pipeline.showBackfillModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showBackfillModal = false"></div>
		<div class="relative w-full max-w-md bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground">Backfill medallion tiers</h2>
				<button @click="pipeline.showBackfillModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
			<p class="text-sm text-foreground/60">Rebuild lake objects from stored sensor events. Backfill uses deterministic keys per event.</p>
			<label class="space-y-1 block">
				<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Stage</span>
				<select v-model="pipeline.backfillForm.stage" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
					<option value="all">All tiers</option>
					<option value="raw">Raw</option>
					<option value="cleaned">Cleaned</option>
					<option value="business">Business</option>
				</select>
			</label>
			<label class="space-y-1 block">
				<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Max events</span>
				<input v-model.number="pipeline.backfillForm.limit" type="number" min="1" max="5000" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
			</label>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showBackfillModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button
					@click="pipeline.backfillDataLakeStages(); pipeline.showBackfillModal = false"
					:disabled="pipeline.saving || !pipeline.linkedDataLake"
					class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium disabled:opacity-40"
				>
					Run backfill
				</button>
			</div>
		</div>
	</div>

	<div v-if="pipeline.showExportTargetModal" class="fixed inset-0 z-[100] flex items-center justify-center p-4">
		<div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="pipeline.showExportTargetModal = false"></div>
		<div class="relative w-full max-w-2xl bg-card border border-border rounded-xl shadow-2xl p-6 space-y-5 max-h-[90vh] overflow-y-auto">
			<div class="flex justify-between items-center">
				<div>
					<h2 class="text-xl font-bold text-foreground">{{ pipeline.editingExportTargetId ? 'Edit export target' : 'Add export target' }}</h2>
					<p class="text-xs text-foreground/50 mt-1">
						{{ pipeline.editingExportTargetId ? 'Saving updates the existing target.' : 'Creates a new saved export target.' }}
					</p>
				</div>
				<button @click="pipeline.showExportTargetModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
			<div class="grid gap-3 sm:grid-cols-2">
				<label class="space-y-1 sm:col-span-2">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Name</span>
					<input v-model="pipeline.exportForm.targetName" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Stage</span>
					<select v-model="pipeline.exportForm.stage" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
						<option value="raw">Raw</option>
						<option value="cleaned">Cleaned</option>
						<option value="business">Business</option>
					</select>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Adapter</span>
					<select v-model="pipeline.exportForm.adapterType" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
						<option v-for="adapter in pipeline.exportAdapters" :key="adapter.type" :value="adapter.type">{{ adapter.name }}</option>
					</select>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Batch rows</span>
					<input v-model.number="pipeline.exportForm.limit" type="number" min="1" max="500" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Cadence (s)</span>
					<input v-model.number="pipeline.exportForm.cadenceSeconds" type="number" min="15" max="3600" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
				</label>
			</div>
			<div class="flex flex-wrap gap-3">
				<label class="flex items-center gap-2 text-sm text-foreground/70">
					<input v-model="pipeline.exportForm.isContinuous" type="checkbox" class="accent-primary" /> Continuous export
				</label>
				<label v-if="pipeline.supportsSavedCredentials" class="flex items-center gap-2 text-sm text-foreground/70">
					<input v-model="pipeline.exportForm.saveCredentials" type="checkbox" class="accent-primary" /> Save credentials
				</label>
			</div>
			<details class="rounded-lg border border-border bg-surface-2 p-3">
				<summary class="text-sm font-medium text-foreground cursor-pointer">Adapter connection settings</summary>
				<div class="mt-3 space-y-3 text-sm">
					<template v-if="pipeline.exportForm.adapterType === 'postgres'">
						<div class="grid gap-3 sm:grid-cols-2">
							<input v-model="pipeline.exportSettingsForm.postgresHost" placeholder="Host" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model.number="pipeline.exportSettingsForm.postgresPort" type="number" placeholder="Port" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model="pipeline.exportSettingsForm.postgresUsername" placeholder="Username" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model="pipeline.exportSettingsForm.postgresPassword" type="password" placeholder="Password" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model="pipeline.exportSettingsForm.postgresDatabase" placeholder="Database" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model="pipeline.exportSettingsForm.postgresTable" placeholder="Table" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
						</div>
						<select v-model="pipeline.exportSettingsForm.postgresIfExists" class="w-full bg-card border border-border rounded-lg px-3 py-2">
							<option value="append">Append rows</option>
							<option value="replace">Replace table</option>
						</select>
					</template>
					<template v-else-if="pipeline.exportForm.adapterType === 'mongodb'">
						<input v-model="pipeline.exportSettingsForm.mongodbUri" placeholder="Mongo URI" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
						<div class="grid gap-3 sm:grid-cols-2">
							<input v-model="pipeline.exportSettingsForm.mongodbDatabase" placeholder="Database" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
							<input v-model="pipeline.exportSettingsForm.mongodbCollection" placeholder="Collection" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
						</div>
						<select v-model="pipeline.exportSettingsForm.mongodbIfExists" class="w-full bg-card border border-border rounded-lg px-3 py-2">
							<option value="append">Append documents</option>
							<option value="replace">Replace collection</option>
						</select>
					</template>
					<template v-else-if="pipeline.exportForm.adapterType === 'huggingface'">
						<div v-if="pipeline.usesSavedHuggingFaceToken" class="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs text-foreground/70">
							Using saved Hugging Face token from Settings.
						</div>
						<input v-else v-model="pipeline.exportSettingsForm.huggingFaceToken" type="password" placeholder="hf_..." class="w-full bg-card border border-border rounded-lg px-3 py-2" />
						<input v-model="pipeline.exportSettingsForm.huggingFaceRepoName" placeholder="Dataset repo name" class="w-full bg-card border border-border rounded-lg px-3 py-2" />
						<p class="text-xs text-foreground/55 leading-relaxed">
							Rows are exported as hourly UTC CSV parts under <span class="font-mono text-foreground/70">data/hourly/YYYY-MM-DDTHH.csv</span>.
							The dataset README and manifest describe each hour file and total row counts.
						</p>
					</template>
					<p v-else class="text-foreground/50 text-xs">{{ pipeline.exportSettingsPreview }}</p>
				</div>
			</details>
			<div class="flex flex-wrap justify-end gap-2">
				<button
					v-if="pipeline.editingExportTargetId"
					@click="pipeline.deleteEditingExportTarget()"
					class="px-4 py-2 rounded-lg border border-red-500/30 text-red-400 hover:text-red-300 mr-auto"
				>
					Delete target
				</button>
				<button @click="pipeline.showExportTargetModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button
					@click="pipeline.saveExportTarget().then(() => { if (!pipeline.saving) pipeline.showExportTargetModal = false })"
					:disabled="pipeline.saving"
					class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium disabled:opacity-40"
				>
					{{ pipeline.editingExportTargetId ? 'Update target' : 'Save target' }}
				</button>
				<button
					v-if="!pipeline.editingExportTargetId"
					@click="pipeline.exportStageToAdapter().then(() => { pipeline.showExportTargetModal = false })"
					:disabled="pipeline.saving"
					class="px-4 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Export once
				</button>
			</div>
		</div>
	</div>
	</Teleport>
</template>
