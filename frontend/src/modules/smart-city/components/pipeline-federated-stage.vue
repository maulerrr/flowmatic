<script setup lang="ts">
import { inject } from 'vue'
import { Network } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section v-show="pipeline.activeStage === 'federated'" class="pipeline-panel pipeline-panel__body space-y-5">
		<div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
			<div class="max-w-2xl">
				<p class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Step 4</p>
				<h3 class="mt-1 text-lg font-semibold text-foreground flex items-center gap-2">
					<Network class="w-4 h-4" /> Federated Model Training
				</h3>
				<p class="mt-2 text-sm text-foreground/60">
					This stage connects the pipeline to a federated learning coordinator and forwards processing and training updates there. It does not control which model preprocesses live events.
				</p>
			</div>
			<div class="flex flex-wrap gap-2">
				<button @click="pipeline.openFederatedWorkspace" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground">
					Open workflow graph
				</button>
			</div>
		</div>
		<div class="grid gap-5 xl:grid-cols-[0.95fr,1.05fr]">
			<div class="space-y-3">
				<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-3">
					<div>
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Coordinator connection</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.federatedConfig.status }}</div>
						<div class="text-xs text-foreground/50 break-words">{{ pipeline.federatedConfig.endpoint || 'No endpoint configured' }}</div>
					</div>
					<div class="grid gap-3 sm:grid-cols-2">
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Protocol</span>
							<select v-model="pipeline.federatedForm.protocol" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground">
								<option value="HTTP">HTTP</option>
								<option value="WEBSOCKET">WebSocket</option>
							</select>
						</label>
						<label class="space-y-1 sm:col-span-2">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Endpoint</span>
							<input
								v-model="pipeline.federatedForm.endpoint"
								type="text"
								:placeholder="pipeline.demoFederatedEndpoint"
								class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground"
							/>
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Project ID</span>
							<input v-model="pipeline.federatedForm.projectId" type="text" placeholder="astana-q1" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Node ID</span>
							<input v-model="pipeline.federatedForm.nodeId" type="text" placeholder="flowmatic-node-1" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Topic</span>
							<input v-model="pipeline.federatedForm.topic" type="text" placeholder="smart-city.training" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">API Key</span>
							<input v-model="pipeline.federatedForm.apiKey" type="password" placeholder="Optional bearer token" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
					</div>
					<div class="grid grid-cols-3 gap-2">
						<button @click="pipeline.connectFederated" :disabled="pipeline.saving" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50">
							Connect
						</button>
						<button @click="pipeline.testFederatedConnection" :disabled="!pipeline.selectedPipelineId" class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40">
							Test
						</button>
						<button
							@click="pipeline.disconnectFederatedConnection"
							:disabled="pipeline.federatedConfig.status === 'DISCONNECTED'"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
						>
							Disconnect
						</button>
					</div>
				</div>
				<div class="grid gap-3 sm:grid-cols-2">
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Last coordinator test</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.federatedConfig.lastTestResult || 'Not tested yet' }}</div>
						<div class="text-xs text-foreground/50">{{ pipeline.formatDateTime(pipeline.federatedConfig.lastTestedAt) }}</div>
					</div>
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Last error</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.federatedConfig.lastError || 'No connection errors' }}</div>
						<div class="text-xs text-foreground/50">
							{{
								pipeline.federatedConfig.lastConnectedAt
									? `Connected ${pipeline.formatDateTime(pipeline.federatedConfig.lastConnectedAt)}`
									: 'No successful connection yet'
							}}
						</div>
					</div>
				</div>
				<div class="grid gap-3 sm:grid-cols-2">
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Registration</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.federatedConfig.registrationId || 'Not registered' }}</div>
						<div class="text-xs text-foreground/50">
							{{
								pipeline.federatedConfig.registeredAt
									? `Registered ${pipeline.formatDateTime(pipeline.federatedConfig.registeredAt)}`
									: 'Connection exists but coordinator did not return an id'
							}}
						</div>
					</div>
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Global model</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.federatedConfig.globalModelVersion || 'No aggregated model yet' }}</div>
						<div class="text-xs text-foreground/50">
							{{
								pipeline.federatedConfig.lastDeliveryAt
									? `Last federated delivery ${pipeline.formatDateTime(pipeline.federatedConfig.lastDeliveryAt)}`
									: 'No round updates delivered yet'
							}}
						</div>
					</div>
				</div>
			</div>
			<div class="space-y-3">
				<div class="rounded-lg border border-border bg-surface-2 p-4 space-y-4">
					<div class="flex items-center justify-between">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Round protocol</div>
						<button
							@click="pipeline.syncFederatedGlobalState"
							:disabled="!pipeline.selectedPipelineId || pipeline.federatedConfig.status === 'DISCONNECTED'"
							class="px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40"
						>
							Sync global model
						</button>
					</div>
					<div class="grid gap-3 sm:grid-cols-2">
						<div class="rounded-lg border border-border bg-card p-3">
							<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Active round</div>
							<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.activeFederatedRound?.name || 'No active round' }}</div>
							<div class="text-xs text-foreground/50">
								{{
									pipeline.activeFederatedRound
										? `${pipeline.activeFederatedRound.status} / ${pipeline.activeFederatedRound.participants.length} participant updates`
										: 'Start a round to collect local updates'
								}}
							</div>
						</div>
						<div class="rounded-lg border border-border bg-card p-3">
							<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Current round id</div>
							<div class="mt-2 text-sm font-semibold text-foreground break-all">{{ pipeline.federatedConfig.currentRoundId || 'None' }}</div>
							<div class="text-xs text-foreground/50">{{ pipeline.federatedRounds.length }} stored rounds</div>
						</div>
					</div>
					<div class="grid gap-3 sm:grid-cols-2">
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Round name</span>
							<input v-model="pipeline.federatedRoundForm.name" type="text" placeholder="Astana round 1" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Sample count</span>
							<input v-model.number="pipeline.federatedRoundForm.sampleCount" type="number" min="1" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
					</div>
					<button
						@click="pipeline.startFederatedRoundFlow"
						:disabled="!pipeline.selectedPipelineId || pipeline.federatedConfig.status === 'DISCONNECTED' || !!pipeline.activeFederatedRound"
						class="w-full px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-40"
					>
						Start round
					</button>
					<div class="grid gap-3 sm:grid-cols-2">
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Checkpoint URI</span>
							<input v-model="pipeline.federatedUpdateForm.checkpointUri" type="text" placeholder="s3://models/local-update.pt" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Update samples</span>
							<input v-model.number="pipeline.federatedUpdateForm.sampleCount" type="number" min="1" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1 sm:col-span-2">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Update notes</span>
							<input
								v-model="pipeline.federatedUpdateForm.notes"
								type="text"
								placeholder="Local fine-tune from business stage exports"
								class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground"
							/>
						</label>
					</div>
					<button @click="pipeline.submitFederatedRoundUpdate" :disabled="!pipeline.activeFederatedRound" class="w-full px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40">
						Submit local update
					</button>
					<div class="grid gap-3 sm:grid-cols-2">
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Global model version</span>
							<input v-model="pipeline.federatedAggregateForm.globalModelVersion" type="text" placeholder="global-20260422-01" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Aggregated checkpoint</span>
							<input v-model="pipeline.federatedAggregateForm.checkpointUri" type="text" placeholder="s3://models/global-20260422-01.pt" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
						<label class="space-y-1 sm:col-span-2">
							<span class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Aggregation summary</span>
							<input v-model="pipeline.federatedAggregateForm.summary" type="text" placeholder="FedAvg completed from 4 participants" class="w-full bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground" />
						</label>
					</div>
					<button @click="pipeline.aggregateFederatedRoundFlow" :disabled="!pipeline.activeFederatedRound" class="w-full px-3 py-2 rounded-lg border border-border text-sm text-foreground/70 hover:text-foreground disabled:opacity-40">
						Aggregate round
					</button>
				</div>
				<div class="flex items-center justify-between">
					<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Local training fallback</div>
					<button @click="pipeline.trainAstanaModel" :disabled="pipeline.saving" class="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50">
						Train baseline
					</button>
				</div>
				<div class="grid gap-3 sm:grid-cols-2">
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Latest training run</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.latestTrainingRun?.status ?? 'Idle' }}</div>
						<div class="text-xs text-foreground/50">{{ pipeline.latestTrainingRun?.name ?? 'No training started yet' }}</div>
					</div>
					<div class="rounded-lg border border-border bg-surface-2 p-4">
						<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Workflow graph</div>
						<div class="mt-2 text-sm font-semibold text-foreground">{{ pipeline.n8nWorkflow?.nodes?.length ?? 0 }} nodes</div>
						<div class="text-xs text-foreground/50">Source -> processing -> lake -> federated coordinator</div>
					</div>
				</div>
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Round ledger</div>
				<div v-if="pipeline.federatedRounds.length === 0" class="rounded-lg border border-dashed border-border p-6 text-sm text-foreground/50">
					No federated rounds yet.
				</div>
				<div v-for="round in pipeline.federatedRounds.slice(0, 4)" :key="round.id" class="rounded-lg border border-border bg-surface-2 p-3 text-xs space-y-1">
					<div class="flex items-start justify-between gap-3">
						<div class="font-semibold text-foreground">{{ round.name }}</div>
						<div class="text-foreground/50">{{ round.status }}</div>
					</div>
					<div class="text-foreground/50 font-mono break-all">{{ round.id }}</div>
					<div class="text-foreground/40">{{ round.participants.length }} participant updates / {{ round.aggregatedModelVersion || 'No global model yet' }}</div>
					<div class="text-foreground/40">
						Started {{ pipeline.formatDateTime(round.startedAt) }}{{ round.completedAt ? ` / Completed ${pipeline.formatDateTime(round.completedAt)}` : '' }}
					</div>
				</div>
				<div class="text-[10px] font-bold uppercase tracking-wider text-foreground/40">Recent training runs</div>
				<div v-if="pipeline.trainingRuns.length === 0" class="rounded-lg border border-dashed border-border p-6 text-sm text-foreground/50">
					No training runs yet.
				</div>
				<div v-for="run in pipeline.trainingRuns.slice(0, 3)" :key="run.id" class="rounded-lg border border-border bg-surface-2 p-3 text-xs">
					<div class="font-semibold text-foreground">{{ run.name }}</div>
					<div class="text-foreground/50 font-mono">{{ run.modelType }} / {{ run.status }}</div>
					<div class="text-foreground/40 mt-1">{{ run.datasetPath }}</div>
				</div>
			</div>
		</div>
	</section>
</template>
