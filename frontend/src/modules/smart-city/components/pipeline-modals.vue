<script setup lang="ts">
import { inject } from 'vue'
import { Cpu, Save, X } from 'lucide-vue-next'
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
			</select>
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
		<div class="relative w-full max-w-lg bg-card border border-border rounded-xl shadow-2xl p-6 space-y-6">
			<div class="flex justify-between items-center">
				<h2 class="text-xl font-bold text-foreground flex items-center gap-2"><Cpu class="w-5 h-5 text-primary" /> Processing Configuration</h2>
				<button @click="pipeline.showConfigModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
			</div>
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
			<div>
				<div class="flex justify-between text-sm mb-2">
					<span class="text-foreground font-medium">Throughput Limit</span>
					<span class="font-mono text-primary">{{ pipeline.streamConfig.throughputLimit }} GB/s</span>
				</div>
				<input v-model.number="pipeline.streamConfig.throughputLimit" type="range" min="1" max="20" step="0.5" class="w-full accent-primary" />
			</div>
			<div class="flex justify-end gap-3">
				<button @click="pipeline.showConfigModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2">Cancel</button>
				<button @click="pipeline.saveConfiguration" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium flex items-center gap-2">
					<Save class="w-4 h-4" /> Apply
				</button>
			</div>
		</div>
	</div>
	</Teleport>
</template>
