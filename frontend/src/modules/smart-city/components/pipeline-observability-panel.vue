<script setup lang="ts">
import { inject } from 'vue'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-if="pipeline.observability" class="pipeline-panel pipeline-panel__body space-y-4">
		<div>
			<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Operations snapshot</p>
			<p class="mt-1 text-sm text-foreground/60">Recent health and alert signals for this pipeline.</p>
		</div>
		<div class="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Events 24h</div>
				<div class="mt-2 text-lg font-semibold text-foreground">{{ pipeline.observability.eventsLast24h }}</div>
				<div class="text-xs text-foreground/50">
					{{
						pipeline.observability.lastEventAt
							? `Last event ${pipeline.formatDateTime(pipeline.observability.lastEventAt)}`
							: 'No recent events'
					}}
				</div>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Sources</div>
				<div class="mt-2 text-lg font-semibold text-foreground">{{ pipeline.observability.runningSources }} running</div>
				<div class="text-xs text-foreground/50">{{ pipeline.observability.sourceErrors }} sources in error</div>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Exports</div>
				<div class="mt-2 text-lg font-semibold text-foreground">
					{{ pipeline.observability.exportSuccesses }} ok / {{ pipeline.observability.exportFailures }} failed
				</div>
				<div class="text-xs text-foreground/50">
					{{
						pipeline.observability.lastExportAt
							? `Last export ${pipeline.formatDateTime(pipeline.observability.lastExportAt)}`
							: 'No recent exports'
					}}
				</div>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Federated</div>
				<div class="mt-2 text-lg font-semibold text-foreground">{{ pipeline.observability.federatedStatus }}</div>
				<div class="text-xs text-foreground/50">{{ pipeline.observability.globalModelVersion || 'No global model version yet' }}</div>
			</div>
		</div>
		<div class="grid gap-2">
			<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Alerts</div>
			<div v-if="pipeline.observability.alerts.length === 0" class="rounded-lg border border-border bg-surface-2 p-3 text-sm text-foreground/50">
				No active alerts.
			</div>
			<div
				v-for="(alert, index) in pipeline.observability.alerts"
				:key="index"
				class="rounded-lg border border-border bg-surface-2 p-3 text-sm"
			>
				<div class="font-semibold text-foreground">{{ alert.severity.toUpperCase() }}</div>
				<div class="text-foreground/60">{{ alert.message }}</div>
			</div>
		</div>
	</section>
</template>
