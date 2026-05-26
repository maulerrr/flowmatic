<script setup lang="ts">
import { inject } from 'vue'
import { Database, HardDrive, MoreHorizontal, Pause, Play, RotateCcw, Trash2 } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')

const lakeTabs = [
	{ id: 'overview' as const, label: 'Overview' },
	{ id: 'browser' as const, label: 'Lake browser' },
	{ id: 'export' as const, label: 'Export targets' },
]
</script>

<template>
	<section v-show="pipeline.activeStage === 'lake'" class="pipeline-panel pipeline-panel__body space-y-5">
		<div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
			<div class="max-w-2xl">
				<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Stage 03</p>
				<h3 class="mt-1 text-lg font-semibold text-foreground flex items-center gap-2">
					<Database class="w-4 h-4" /> Data Lake & Export
				</h3>
				<p class="mt-2 text-sm text-foreground/60">
					Medallion lake writes and static export targets. Realtime events append to daily <code class="text-xs">stream.ndjson</code> files when append mode is enabled.
				</p>
			</div>
			<button @click="pipeline.showDataLakeModal = true" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium">
				Connect lake
			</button>
		</div>

		<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-4">
			<div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Pipeline runtime</div>
					<div class="mt-1 text-sm font-semibold text-foreground">{{ pipeline.pipelineRuntimeLabel }}</div>
				</div>
				<div class="flex flex-wrap gap-2">
					<button
						v-if="pipeline.pipelineRuntimeStatus !== 'ACTIVE'"
						@click="pipeline.startPipelineRuntime"
						:disabled="pipeline.saving || !pipeline.selectedPipelineId"
						class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium flex items-center gap-1.5 disabled:opacity-40"
					>
						<Play class="w-3.5 h-3.5" /> Start
					</button>
					<button
						v-if="pipeline.pipelineRuntimeStatus === 'ACTIVE'"
						@click="pipeline.stopPipelineRuntime"
						:disabled="pipeline.saving"
						class="px-3 py-2 rounded-lg border border-warning/30 bg-warning/10 text-warning text-xs font-medium flex items-center gap-1.5"
					>
						<Pause class="w-3.5 h-3.5" /> Stop
					</button>
					<button
						v-if="pipeline.pipelineRuntimeStatus === 'PAUSED'"
						@click="pipeline.resumePipelineRuntime"
						:disabled="pipeline.saving"
						class="px-3 py-2 rounded-lg border border-border text-xs font-medium flex items-center gap-1.5 text-foreground/70 hover:text-foreground"
					>
						<RotateCcw class="w-3.5 h-3.5" /> Resume
					</button>
				</div>
			</div>
			<div class="grid gap-3 sm:grid-cols-3">
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Source interval (ms)</span>
					<input
						v-model.number="pipeline.runtimeForm.sourcePollIntervalMs"
						type="number"
						min="1000"
						step="1000"
						class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground"
					/>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Lake write mode</span>
					<select v-model="pipeline.runtimeForm.lakeWriteMode" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground">
						<option value="append">Append stream.ndjson</option>
						<option value="object">One object per event</option>
					</select>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Export cadence (s)</span>
					<input
						v-model.number="pipeline.runtimeForm.exportCadenceSeconds"
						type="number"
						min="15"
						max="3600"
						class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground"
					/>
				</label>
			</div>
			<div class="flex justify-end">
				<button
					@click="pipeline.savePipelineRuntime"
					:disabled="pipeline.saving || !pipeline.selectedPipelineId"
					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Save runtime settings
				</button>
			</div>
		</div>

		<div class="flex flex-wrap gap-2 border-b border-border pb-2">
			<button
				v-for="tab in lakeTabs"
				:key="tab.id"
				@click="pipeline.lakeExportTab = tab.id"
				:class="[
					'px-3 py-1.5 rounded-lg text-sm font-medium',
					pipeline.lakeExportTab === tab.id ? 'bg-primary/10 text-primary' : 'text-foreground/60 hover:text-foreground',
				]"
			>
				{{ tab.label }}
			</button>
		</div>

		<div v-if="pipeline.lakeExportTab === 'overview'" class="grid gap-4 lg:grid-cols-[1fr,1fr]">
			<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
				<div class="flex items-center gap-3">
					<HardDrive class="w-5 h-5 text-blue-500" />
					<div class="min-w-0">
						<p class="font-semibold text-foreground text-sm">{{ pipeline.linkedDataLake?.name ?? 'No lake linked' }}</p>
						<p class="text-xs text-foreground/50 truncate">{{ pipeline.linkedDataLake?.bucket ?? 'Connect MinIO or S3' }}</p>
					</div>
				</div>
				<p class="text-xs text-foreground/50">
					Linked lake receives raw, cleaned, and business tiers under
					<span class="font-mono">{{ pipeline.linkedDataLake?.basePrefix || 'your prefix' }}</span>.
				</p>
				<div v-if="pipeline.linkedDataLake" class="flex flex-wrap gap-2">
					<button @click="pipeline.testDataLake(pipeline.linkedDataLake)" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground">Test</button>
					<button @click="pipeline.disconnectDataLake(pipeline.linkedDataLake)" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground">Disconnect</button>
					<button @click="pipeline.showBackfillModal = true" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground">Backfill…</button>
				</div>
			</div>
			<div class="rounded-lg border border-border bg-surface-2 p-4 grid gap-3 sm:grid-cols-2 text-sm">
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Status</div>
					<div class="mt-1 font-semibold">{{ pipeline.linkedDataLake?.lastTestStatus ?? 'Not tested' }}</div>
				</div>
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Layout</div>
					<div class="mt-1 font-semibold">raw / cleaned / business</div>
				</div>
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Append mode</div>
					<div class="mt-1 font-semibold">{{ pipeline.runtimeForm.lakeWriteMode === 'append' ? 'stream.ndjson per hour' : 'Per-event JSON' }}</div>
				</div>
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Last test</div>
					<div class="mt-1 text-xs text-foreground/60">{{ pipeline.formatDateTime(pipeline.linkedDataLake?.lastTestedAt) }}</div>
				</div>
			</div>
		</div>

		<div v-else-if="pipeline.lakeExportTab === 'browser'" class="rounded-lg border border-border bg-surface-2 p-4 space-y-4">
			<div class="flex items-center justify-between">
				<div class="text-sm text-foreground/60">Objects under today&apos;s day partition for each medallion tier.</div>
				<button @click="pipeline.refreshLakeBrowser" :disabled="!pipeline.selectedPipelineId" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40">
					Refresh
				</button>
			</div>
			<div class="grid gap-3 lg:grid-cols-3">
				<div v-for="group in pipeline.dataLakeObjects" :key="group.stage" class="rounded-lg border border-border bg-card p-3 space-y-2">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">{{ group.stage }}</div>
					<div class="text-xs text-foreground/50 break-words">{{ group.prefix }}</div>
					<div v-if="group.objects.length === 0" class="text-xs text-foreground/40">No objects yet.</div>
					<div v-for="object in group.objects.slice(0, 6)" :key="object.key" class="rounded border border-border px-2 py-2 text-xs">
						<div class="font-medium text-foreground break-all">{{ object.key.split('/').slice(-1)[0] }}</div>
						<div class="text-foreground/50">{{ pipeline.formatBytes(object.size) }}</div>
					</div>
				</div>
			</div>
		</div>

		<div v-else class="space-y-4">
			<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
				<div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Export data preview</div>
						<p class="text-xs text-foreground/50 mt-1">Sample rows that continuous exports will send (latest events for the selected stage).</p>
					</div>
					<div class="flex items-center gap-2">
						<select
							v-model="pipeline.exportPreviewStage"
							class="bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground"
						>
							<option value="raw">Raw</option>
							<option value="cleaned">Cleaned</option>
							<option value="business">Business</option>
						</select>
						<button
							@click="pipeline.loadExportPreview()"
							:disabled="!pipeline.selectedPipelineId"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
						>
							Refresh
						</button>
					</div>
				</div>
				<div v-if="!pipeline.exportPreview || pipeline.exportPreview.rows.length === 0" class="text-sm text-foreground/50">
					No rows yet for this stage. Start the pipeline and wait for sensor events.
				</div>
				<div v-else class="overflow-x-auto">
					<table class="min-w-full text-xs">
						<thead>
							<tr class="text-left text-foreground/50 border-b border-border">
								<th v-for="column in pipeline.exportPreview.columns.slice(0, 8)" :key="column" class="py-2 pr-3 font-medium">{{ column }}</th>
							</tr>
						</thead>
						<tbody>
							<tr v-for="(row, index) in pipeline.exportPreview.rows" :key="index" class="border-b border-border/40">
								<td v-for="column in pipeline.exportPreview.columns.slice(0, 8)" :key="column" class="py-2 pr-3 text-foreground/80 font-mono">
									{{ row[column] ?? '—' }}
								</td>
							</tr>
						</tbody>
					</table>
				</div>
			</div>

			<div class="flex items-center justify-between gap-3">
				<p class="text-sm text-foreground/60">Continuous targets export new rows on a cursor schedule (Postgres/Mongo/HF append when enabled).</p>
				<button @click="pipeline.openExportTargetModal()" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium">
					Add export target
				</button>
			</div>
			<div v-if="pipeline.exportTargets.length === 0" class="rounded-lg border border-dashed border-border p-6 text-sm text-foreground/50 text-center">
				No export targets yet. Add one to push cleaned or business rows to JSON, CSV, Postgres, or MongoDB.
			</div>
			<div v-for="target in pipeline.exportTargets" :key="target.id" class="rounded-lg border border-border bg-card p-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
				<div class="min-w-0 flex-1">
					<div class="font-semibold text-foreground">{{ target.name }}</div>
					<div class="text-xs text-foreground/50 font-mono">{{ target.stage }} · {{ target.adapterType }} · {{ target.status }}</div>
					<div class="text-xs text-foreground/40 mt-1">
						{{ target.isContinuous ? `Every ${target.cadenceSeconds}s` : 'Manual' }}
						<span v-if="target.lastRunAt"> · last {{ pipeline.formatDateTime(target.lastRunAt) }}</span>
					</div>
					<div v-if="target.lastError" class="text-xs text-red-400 mt-1 break-words">{{ target.lastError }}</div>
				</div>
				<div class="flex flex-wrap items-center gap-2 shrink-0">
					<button @click="pipeline.openExportTargetModal(target)" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground inline-flex items-center gap-1">
						<MoreHorizontal class="w-3.5 h-3.5" /> Edit
					</button>
					<button @click="pipeline.runExportTarget(target)" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground">Run</button>
					<button @click="pipeline.toggleContinuousTarget(target)" class="px-2 py-1.5 rounded-lg border border-border text-xs text-foreground/70 hover:text-foreground">
						{{ target.isContinuous ? 'Pause' : 'Enable' }}
					</button>
					<button
						@click="pipeline.deleteExportTarget(target)"
						class="px-2 py-1.5 rounded-lg border border-red-500/30 bg-red-500/10 text-xs text-red-400 hover:text-red-300 inline-flex items-center gap-1"
						title="Delete export target"
					>
						<Trash2 class="w-3.5 h-3.5" /> Delete
					</button>
				</div>
			</div>
			<div v-if="pipeline.exportRuns.length > 0" class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Recent runs</div>
				<div v-for="run in pipeline.exportRuns.slice(0, 4)" :key="run.id" class="text-xs border-b border-border/50 pb-3 last:border-0 space-y-2">
					<div class="flex justify-between gap-2">
						<span>{{ run.stage }} → {{ run.adapterType }} ({{ run.status === 'FAILED' ? run.rowCount : run.recordsExported }} rows)</span>
						<span :class="run.status === 'SUCCEEDED' ? 'text-success' : run.status === 'FAILED' ? 'text-red-400' : 'text-foreground/50'">{{ run.status }}</span>
					</div>
					<div v-if="run.destination" class="text-foreground/50 break-all">{{ run.destination }}</div>
					<div v-if="Array.isArray(run.metadata?.previewRows) && run.metadata.previewRows.length" class="overflow-x-auto">
						<table class="min-w-full text-[10px]">
							<thead>
								<tr class="text-foreground/40">
									<th v-for="col in (run.metadata.previewColumns as string[] | undefined)?.slice(0, 6) ?? []" :key="col" class="pr-2 py-1 text-left">{{ col }}</th>
								</tr>
							</thead>
							<tbody>
								<tr v-for="(row, idx) in (run.metadata.previewRows as Record<string, unknown>[]).slice(0, 3)" :key="idx">
									<td v-for="col in (run.metadata.previewColumns as string[] | undefined)?.slice(0, 6) ?? []" :key="col" class="pr-2 py-1 font-mono text-foreground/70">
										{{ (row as Record<string, unknown>)[col] ?? '—' }}
									</td>
								</tr>
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
	</section>
</template>
