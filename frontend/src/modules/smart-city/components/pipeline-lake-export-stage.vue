<script setup lang="ts">
import { inject } from 'vue'
import { Database, HardDrive } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
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
					This stage stores medallion data in the lake and can also export pipeline stages into static destinations like Hugging Face, PostgreSQL, or MongoDB.
				</p>
			</div>
			<button @click="pipeline.showDataLakeModal = true" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium">
				Connect lake
			</button>
		</div>
		<div class="grid gap-5 xl:grid-cols-[0.9fr,1.1fr]">
			<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
				<div class="flex items-center gap-3">
					<HardDrive class="w-5 h-5 text-blue-500" />
					<div class="min-w-0">
						<p class="font-semibold text-foreground text-sm">{{ pipeline.activeDataLake?.name ?? 'Not connected' }}</p>
						<p class="text-xs text-foreground/50 truncate">{{ pipeline.activeDataLake?.bucket ?? 'Create an S3-compatible target' }}</p>
					</div>
				</div>
				<div v-if="pipeline.activeDataLake" class="grid grid-cols-3 gap-2">
					<button
						@click="pipeline.testDataLake(pipeline.activeDataLake)"
						class="px-2 py-2 rounded-lg border border-border text-foreground/70 text-xs font-medium hover:text-foreground"
					>
						Test
					</button>
					<button
						@click="pipeline.disconnectDataLake(pipeline.activeDataLake)"
						class="px-2 py-2 rounded-lg border border-border text-foreground/70 text-xs font-medium hover:text-foreground"
					>
						Disconnect
					</button>
					<button
						@click="pipeline.deleteDataLakeConnection(pipeline.activeDataLake)"
						class="px-2 py-2 rounded-lg border border-red-500/20 text-red-400 text-xs font-medium"
					>
						Delete
					</button>
				</div>
			</div>
			<div class="grid gap-3 sm:grid-cols-2">
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Linked lake</div>
					<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.linkedDataLake?.name ?? 'No lake linked to this pipeline' }}</div>
					<div class="text-xs text-foreground/50">{{ pipeline.linkedDataLake?.provider ?? 'Link a lake to route output' }}</div>
				</div>
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Output path</div>
					<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.linkedDataLake?.basePrefix || 'No prefix configured' }}</div>
					<div class="text-xs text-foreground/50">{{ pipeline.linkedDataLake ? 'Pipeline output is bucketed under this prefix' : 'Configure storage first' }}</div>
				</div>
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Last connection test</div>
					<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.activeDataLake?.lastTestStatus ?? 'Not tested yet' }}</div>
					<div class="text-xs text-foreground/50">{{ pipeline.formatDateTime(pipeline.activeDataLake?.lastTestedAt) }}</div>
				</div>
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">What gets stored</div>
					<div class="mt-2 text-sm font-semibold text-foreground">Raw, cleaned, business tiers</div>
					<div class="text-xs text-foreground/50">Raw events, cleaned records, and business-level outputs are written separately.</div>
				</div>
				<div class="rounded-lg border border-border bg-surface-2 p-4">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Medallion layout</div>
					<div class="mt-2 text-sm font-semibold text-foreground">raw / cleaned / business</div>
					<div class="text-xs text-foreground/50">The lake now follows medallion tiers instead of one generic output folder.</div>
				</div>
			</div>
		</div>
		<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-4">
			<div class="flex items-center justify-between">
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Lake browser</div>
					<div class="mt-1 text-sm text-foreground/60">Browse the latest written objects by medallion stage.</div>
				</div>
				<button @click="pipeline.refreshLakeBrowser" :disabled="!pipeline.selectedPipelineId" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40">
					Refresh browser
				</button>
			</div>
			<div class="grid gap-3 lg:grid-cols-3">
				<div v-for="group in pipeline.dataLakeObjects" :key="group.stage" class="rounded-lg border border-border bg-card p-3 space-y-2">
					<div>
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">{{ group.stage }}</div>
						<div class="mt-1 text-xs text-foreground/50 break-words">{{ group.prefix }}</div>
					</div>
					<div v-if="group.objects.length === 0" class="text-xs text-foreground/40">No objects yet.</div>
					<div v-for="object in group.objects.slice(0, 5)" :key="object.key" class="rounded border border-border px-2 py-2 text-xs">
						<div class="font-medium text-foreground break-all">{{ object.key.split('/').slice(-1)[0] }}</div>
						<div class="text-foreground/50">{{ pipeline.formatBytes(object.size) }} / {{ pipeline.formatDateTime(object.lastModified) }}</div>
					</div>
				</div>
			</div>
			<div class="rounded-lg border border-border bg-card p-4 space-y-4">
				<div class="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
					<div>
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Replay and backfill</div>
						<div class="mt-1 text-sm text-foreground/60">Rebuild medallion outputs from historical sensor events already stored in Flowmatic.</div>
					</div>
					<div class="text-xs text-foreground/40">
						Backfill writes deterministic objects, so rerunning refreshes the same historical slices.
					</div>
				</div>
				<div class="grid gap-3 lg:grid-cols-[180px,180px,1fr]">
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Stage</span>
						<select v-model="pipeline.backfillForm.stage" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
							<option value="all">All medallion stages</option>
							<option value="raw">Raw only</option>
							<option value="cleaned">Cleaned only</option>
							<option value="business">Business only</option>
						</select>
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Events</span>
						<input
							v-model.number="pipeline.backfillForm.limit"
							type="number"
							min="1"
							max="5000"
							class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
						/>
					</label>
					<div class="flex items-end justify-end">
						<button
							@click="pipeline.backfillDataLakeStages"
							:disabled="pipeline.saving || !pipeline.selectedPipelineId || !pipeline.activeDataLake"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
						>
							Run backfill
						</button>
					</div>
				</div>
				<div v-if="pipeline.latestBackfill" class="rounded-lg border border-border bg-surface-2 p-3 text-xs space-y-1">
					<div class="font-semibold text-foreground">
						Last backfill: {{ pipeline.latestBackfill.stage }} / {{ pipeline.latestBackfill.scannedEvents }} events
					</div>
					<div class="text-foreground/50">
						raw {{ pipeline.latestBackfill.counts.raw }} / cleaned {{ pipeline.latestBackfill.counts.cleaned }} / business {{
							pipeline.latestBackfill.counts.business
						}}
					</div>
					<div class="text-foreground/40">{{ pipeline.formatDateTime(pipeline.latestBackfill.completedAt) }}</div>
				</div>
			</div>
		</div>
		<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-4">
			<div class="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
				<div>
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Static export</div>
					<div class="mt-1 text-sm text-foreground/60">Push one medallion stage to an adapter using the same export stack as the file pipeline.</div>
				</div>
				<div class="max-w-md text-right">
					<div class="text-xs text-foreground/50">
						{{ pipeline.selectedExportAdapter?.requiredSettings?.join(', ') || 'Pick an adapter to see required settings' }}
					</div>
					<div class="mt-1 text-xs text-foreground/40">{{ pipeline.exportSettingsPreview }}</div>
				</div>
			</div>
			<div class="grid gap-3 lg:grid-cols-4">
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Target name</span>
					<input v-model="pipeline.exportForm.targetName" type="text" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Stage</span>
					<select v-model="pipeline.exportForm.stage" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground">
						<option value="raw">Raw</option>
						<option value="cleaned">Cleaned</option>
						<option value="business">Business</option>
					</select>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Adapter</span>
					<select v-model="pipeline.exportForm.adapterType" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground">
						<option v-for="adapter in pipeline.exportAdapters" :key="adapter.type" :value="adapter.type">{{ adapter.name }}</option>
					</select>
				</label>
				<label class="space-y-1">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Rows</span>
					<input v-model.number="pipeline.exportForm.limit" type="number" min="1" max="500" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
				</label>
				<div class="grid gap-2">
					<label class="flex items-center gap-2 rounded-lg border border-border px-3 py-2">
						<input v-model="pipeline.exportForm.saveCredentials" type="checkbox" class="accent-primary" />
						<span class="text-sm text-foreground/70">Save credentials</span>
					</label>
					<label class="flex items-center gap-2 rounded-lg border border-border px-3 py-2">
						<input v-model="pipeline.exportForm.isContinuous" type="checkbox" class="accent-primary" />
						<span class="text-sm text-foreground/70">Continuous target</span>
					</label>
				</div>
			</div>
			<div class="rounded-lg border border-border bg-card p-4 space-y-4">
				<div class="flex items-start justify-between gap-3">
					<div>
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Adapter settings</div>
						<div class="mt-1 text-sm text-foreground/60">
							Use the fields below for the selected destination. Advanced JSON stays available for fields not exposed in the form.
						</div>
					</div>
					<div class="text-xs text-foreground/40 max-w-xs text-right">{{ pipeline.exportCredentialHint }}</div>
				</div>

				<div v-if="pipeline.exportForm.adapterType === 'json'" class="grid gap-3 lg:grid-cols-2">
					<label class="flex items-center gap-2 rounded-lg border border-border px-3 py-3">
						<input v-model="pipeline.exportSettingsForm.jsonPrettyPrint" type="checkbox" class="accent-primary" />
						<div>
							<div class="text-sm font-medium text-foreground">Pretty print JSON</div>
							<div class="text-xs text-foreground/50">Write formatted JSON instead of one compact line.</div>
						</div>
					</label>
				</div>

				<div v-else-if="pipeline.exportForm.adapterType === 'csv'" class="grid gap-3 lg:grid-cols-2">
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Delimiter</span>
						<select v-model="pipeline.exportSettingsForm.csvDelimiter" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
							<option value=",">Comma (,)</option>
							<option value=";">Semicolon (;)</option>
							<option value="\t">Tab</option>
							<option value="|">Pipe (|)</option>
						</select>
					</label>
				</div>

				<div v-else-if="pipeline.exportForm.adapterType === 'postgres'" class="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Host</span>
						<input v-model="pipeline.exportSettingsForm.postgresHost" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="localhost" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Port</span>
						<input
							v-model.number="pipeline.exportSettingsForm.postgresPort"
							type="number"
							min="1"
							max="65535"
							class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
						/>
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Database</span>
						<input v-model="pipeline.exportSettingsForm.postgresDatabase" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="flowmatic" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Username</span>
						<input v-model="pipeline.exportSettingsForm.postgresUsername" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="postgres" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Password</span>
						<input
							v-model="pipeline.exportSettingsForm.postgresPassword"
							type="password"
							class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground"
							placeholder="Saved securely if enabled"
						/>
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Table</span>
						<input v-model="pipeline.exportSettingsForm.postgresTable" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="smart_city_business" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">If table exists</span>
						<select v-model="pipeline.exportSettingsForm.postgresIfExists" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
							<option value="append">Append rows</option>
							<option value="replace">Drop and recreate</option>
						</select>
					</label>
				</div>

				<div v-else-if="pipeline.exportForm.adapterType === 'mongodb'" class="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
					<label class="space-y-1 lg:col-span-2">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Connection URI</span>
						<input v-model="pipeline.exportSettingsForm.mongodbUri" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="mongodb://localhost:27017" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Database</span>
						<input v-model="pipeline.exportSettingsForm.mongodbDatabase" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="flowmatic" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Collection</span>
						<input v-model="pipeline.exportSettingsForm.mongodbCollection" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="smart_city_business" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">If collection exists</span>
						<select v-model="pipeline.exportSettingsForm.mongodbIfExists" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground">
							<option value="append">Append documents</option>
							<option value="replace">Clear and replace</option>
						</select>
					</label>
				</div>

				<div v-else-if="pipeline.exportForm.adapterType === 'huggingface'" class="grid gap-3 lg:grid-cols-2 xl:grid-cols-3">
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Token</span>
						<input v-model="pipeline.exportSettingsForm.huggingFaceToken" type="password" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="hf_..." />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Dataset repo name</span>
						<input v-model="pipeline.exportSettingsForm.huggingFaceRepoName" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="smart-city-business" />
					</label>
					<label class="space-y-1">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">File name</span>
						<input v-model="pipeline.exportSettingsForm.huggingFaceFileName" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="smart_city_business.csv" />
					</label>
					<label class="space-y-1 lg:col-span-2">
						<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Commit message</span>
						<input v-model="pipeline.exportSettingsForm.huggingFaceCommitMessage" type="text" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-sm text-foreground" placeholder="Upload business stage from Flowmatic" />
					</label>
					<label class="flex items-center gap-2 rounded-lg border border-border px-3 py-3">
						<input v-model="pipeline.exportSettingsForm.huggingFacePrivate" type="checkbox" class="accent-primary" />
						<div>
							<div class="text-sm font-medium text-foreground">Private dataset</div>
							<div class="text-xs text-foreground/50">Create the repo as private when possible.</div>
						</div>
					</label>
				</div>

				<div class="flex items-center justify-between gap-3 border-t border-border pt-3">
					<div class="text-xs text-foreground/40">Advanced JSON is only for adapter options that are not in the structured form.</div>
					<button @click="pipeline.showAdvancedExportJson = !pipeline.showAdvancedExportJson" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground">
						{{ pipeline.showAdvancedExportJson ? 'Hide advanced JSON' : 'Show advanced JSON' }}
					</button>
				</div>

				<label v-if="pipeline.showAdvancedExportJson" class="space-y-1 block">
					<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Advanced settings JSON</span>
					<textarea
						v-model="pipeline.exportSettingsForm.advancedJson"
						rows="6"
						class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-xs font-mono text-foreground"
						placeholder='{"customOption":"value"}'
					></textarea>
				</label>
			</div>
			<label v-if="pipeline.exportForm.isContinuous" class="space-y-1 block">
				<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Cadence seconds</span>
				<input v-model.number="pipeline.exportForm.cadenceSeconds" type="number" min="15" max="3600" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
			</label>
			<div class="flex justify-end gap-2">
				<button
					@click="pipeline.saveExportTarget"
					:disabled="pipeline.saving || !pipeline.selectedPipelineId"
					class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
				>
					Save target
				</button>
				<button @click="pipeline.exportStageToAdapter" :disabled="pipeline.saving || !pipeline.selectedPipelineId" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-40">
					Export stage
				</button>
			</div>
			<div class="grid gap-4 lg:grid-cols-2">
				<div class="space-y-2">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Saved export targets</div>
					<div v-if="pipeline.exportTargets.length === 0" class="rounded-lg border border-dashed border-border p-4 text-sm text-foreground/50">
						No saved export targets yet.
					</div>
					<div v-for="target in pipeline.exportTargets.slice(0, 6)" :key="target.id" class="rounded-lg border border-border bg-card p-3 text-xs space-y-2">
						<div class="flex items-start justify-between gap-2">
							<div>
								<div class="font-semibold text-foreground">{{ target.name }}</div>
								<div class="text-foreground/50 font-mono">{{ target.stage }} / {{ target.adapterType }} / {{ target.status }}</div>
							</div>
							<span class="text-foreground/40">{{ target.isContinuous ? `${target.cadenceSeconds}s` : 'manual' }}</span>
						</div>
						<div class="text-foreground/40">{{ target.lastRunAt ? `Last run ${pipeline.formatDateTime(target.lastRunAt)}` : 'No runs yet' }}</div>
						<div v-if="target.lastError" class="text-red-400 break-words">{{ target.lastError }}</div>
						<div class="flex gap-2">
							<button @click="pipeline.loadExportTargetIntoForm(target)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">Use</button>
							<button @click="pipeline.runExportTarget(target)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">Run now</button>
							<button @click="pipeline.toggleContinuousTarget(target)" class="px-2 py-1 rounded-lg border border-border text-foreground/70 hover:text-foreground">
								{{ target.isContinuous ? 'Pause' : 'Enable continuous' }}
							</button>
						</div>
					</div>
				</div>
				<div class="space-y-2">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Recent export runs</div>
					<div v-if="pipeline.exportRuns.length === 0" class="rounded-lg border border-dashed border-border p-4 text-sm text-foreground/50">
						No smart-city export runs yet.
					</div>
					<div v-for="run in pipeline.exportRuns.slice(0, 6)" :key="run.id" class="rounded-lg border border-border bg-card p-3 text-xs space-y-1">
						<div class="flex items-start justify-between gap-2">
							<div class="font-semibold text-foreground">{{ run.stage }} -> {{ run.adapterType }}</div>
							<div class="text-foreground/40">{{ run.status }}</div>
						</div>
						<div class="text-foreground/50">{{ run.recordsExported }} exported from {{ run.rowCount }} rows</div>
						<div class="text-foreground/40 break-words">{{ run.destination || run.message || 'Queued' }}</div>
						<div v-if="run.errorMessage" class="text-red-400 break-words">{{ run.errorMessage }}</div>
					</div>
				</div>
			</div>
		</div>
	</section>
</template>
