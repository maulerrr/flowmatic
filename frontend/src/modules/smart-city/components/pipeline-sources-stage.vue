<script setup lang="ts">
import { inject } from 'vue'
import { Radio } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-show="pipeline.activeStage === 'sources'" class="pipeline-panel pipeline-panel__body space-y-5">
		<div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
			<div class="max-w-2xl">
				<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Step 1</p>
				<h3 class="mt-1 text-lg font-semibold text-foreground flex items-center gap-2">
					<Radio class="w-4 h-4" /> Real-Time Sources
				</h3>
				<p class="mt-2 text-sm text-foreground/60">
					These are the inputs of the pipeline. Start a source to stream data continuously, test it to emit one event, or poll the HTTP feeds
					manually.
				</p>
			</div>
			<div class="flex flex-wrap gap-2">
				<button @click="pipeline.showSourceModal = true" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium">
					Add source
				</button>
				<button
					@click="pipeline.pollHttpFeed"
					:disabled="!pipeline.selectedPipelineId"
					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Poll HTTP feeds
				</button>
			</div>
		</div>
		<div class="grid gap-5 xl:grid-cols-[1.4fr,0.8fr]">
			<div class="space-y-3">
				<div v-if="pipeline.sources.length === 0" class="rounded-lg border border-dashed border-border p-6 text-sm text-foreground/50">
					No sources connected yet.
				</div>
				<div v-for="source in pipeline.sources" :key="source.id" class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
					<div class="flex gap-3">
						<component :is="pipeline.sourceIcon(source.sensorKind)" class="w-5 h-5 text-primary mt-1 shrink-0" />
						<div class="min-w-0 flex-1">
							<div class="flex items-center justify-between gap-2">
								<p class="font-semibold text-sm text-foreground truncate">{{ source.name }}</p>
								<span
									:class="[
										'shrink-0 rounded px-2 py-0.5 text-[10px] font-bold uppercase',
										source.status === 'RUNNING' ? 'bg-success/10 text-success' : 'border border-border bg-card text-foreground/50',
									]"
									>{{ source.status }}</span
								>
							</div>
							<p class="text-xs text-foreground/50 font-mono">{{ source.type }} / {{ source.mode }} / {{ source.sensorKind }}</p>
							<p class="text-xs text-foreground/40 break-words">
								{{ source.endpoint || `Poll interval ${source.pollIntervalMs}ms` }}
							</p>
							<p class="text-xs text-foreground/40">
								{{ source.lastSeenAt ? `Last event ${new Date(source.lastSeenAt).toLocaleTimeString()}` : 'No event received yet' }}
							</p>
							<p v-if="source.lastError" class="text-xs text-red-400 break-words">{{ source.lastError }}</p>
						</div>
					</div>
					<div class="grid grid-cols-4 gap-2">
						<button
							v-if="source.status !== 'RUNNING'"
							@click="pipeline.startSource(source)"
							class="px-2 py-2 rounded-lg bg-success/10 text-success border border-success/20 text-xs font-medium"
						>
							Start
						</button>
						<button v-else @click="pipeline.stopSource(source)" class="px-2 py-2 rounded-lg bg-warning/10 text-warning border border-warning/20 text-xs font-medium">
							Stop
						</button>
						<button @click="pipeline.testSource(source)" class="px-2 py-2 rounded-lg border border-border text-foreground/70 text-xs font-medium hover:text-foreground">
							Test
						</button>
						<button @click="pipeline.removeSource(source)" class="px-2 py-2 rounded-lg border border-red-500/20 text-red-400 text-xs font-medium">
							Delete
						</button>
					</div>
				</div>
			</div>
			<div class="space-y-3">
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Source status</p>
					<p class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.sourceStatusText }}</p>
					<p class="text-xs text-foreground/50">
						WebSocket {{ pipeline.wsStatus }}. {{ pipeline.events.length }} recent events captured. {{ pipeline.sourceMixSummary }}.
					</p>
				</div>
				<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
					<div>
						<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Recent events</p>
						<p class="mt-1 text-xs text-foreground/50">Latest events entering the pipeline.</p>
					</div>
					<div v-if="pipeline.recentEventsPreview.length === 0" class="text-sm text-foreground/50">No events received yet.</div>
					<div v-for="event in pipeline.recentEventsPreview" :key="event.id" class="rounded-lg border border-border bg-card px-3 py-2">
						<div class="flex items-center justify-between gap-2">
							<p class="truncate text-sm font-medium text-foreground">{{ event.source?.name ?? event.sensorType }}</p>
							<span class="text-[10px] font-mono text-foreground/40">{{ new Date(event.eventTime).toLocaleTimeString() }}</span>
						</div>
						<p class="text-xs text-foreground/50">{{ event.sensorType }}{{ event.location ? ` / ${event.location}` : '' }}</p>
					</div>
				</div>
			</div>
		</div>
	</section>
</template>
