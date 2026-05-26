<script setup lang="ts">
import { inject } from 'vue'
import { Brain, Cpu, HardDrive, Plus } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section class="pipeline-workflow min-h-[620px] overflow-auto p-8 pipeline-animate-rise">
		<div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between mb-8">
			<div>
				<div class="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/40">Workflow map</div>
				<div class="text-xs font-mono text-foreground/60">
					Generated from real pipeline state: {{ pipeline.n8nWorkflow?.nodes?.length ?? 0 }} nodes
				</div>
			</div>
			<button @click="pipeline.loadN8nWorkflow" class="px-3 py-2 bg-primary/20 text-primary border border-primary/20 rounded-lg text-xs">
				Regenerate
			</button>
		</div>
		<div class="grid lg:grid-cols-3 gap-8 min-w-[900px]">
			<div class="space-y-4">
				<div class="text-[10px] font-bold uppercase tracking-widest text-foreground/30">Data Sources</div>
				<div v-for="source in pipeline.sources" :key="'flow-' + source.id" class="bg-card border border-border rounded-lg p-4">
					<div class="flex items-center gap-3">
						<component :is="pipeline.sourceIcon(source.sensorKind)" class="w-5 h-5 text-primary" />
						<div>
							<div class="text-sm font-bold text-foreground">{{ source.name }}</div>
							<div class="text-[10px] font-mono text-foreground/50">{{ source.type }} / {{ source.status }}</div>
						</div>
					</div>
				</div>
				<button
					@click="pipeline.showSourceModal = true"
					class="w-full border-2 border-dashed border-border rounded-lg p-4 text-foreground/40 hover:text-primary"
				>
					<Plus class="w-5 h-5 mx-auto" />
				</button>
			</div>
			<div class="flex items-center justify-center">
				<div @click="pipeline.openConfigModal()" class="w-full bg-card border-2 border-primary/20 rounded-xl p-5 cursor-pointer hover:border-primary/40 transition-colors">
					<div class="flex items-center gap-2 font-bold text-foreground"><Cpu class="w-5 h-5 text-primary" /> Core unit</div>
					<p class="mt-2 text-xs text-foreground/50 truncate">{{ pipeline.activeModelLabel }}</p>
					<div class="grid grid-cols-2 gap-2 mt-4 text-xs">
						<div class="bg-surface-2 p-3 rounded-lg border border-border">
							Running<br /><b>{{ pipeline.runningSourceCount }}</b>
						</div>
						<div class="bg-surface-2 p-3 rounded-lg border border-border">
							Throughput<br /><b>{{ pipeline.throughput }} GB/s</b>
						</div>
					</div>
				</div>
			</div>
			<div class="space-y-4">
				<div class="bg-card border border-border rounded-lg p-4">
					<div class="flex items-center gap-3">
						<HardDrive class="w-5 h-5 text-blue-500" />
						<div>
							<div class="text-sm font-bold text-foreground">{{ pipeline.activeDataLake?.name ?? 'S3 Data Lake' }}</div>
							<div class="text-[10px] font-mono text-foreground/50">{{ pipeline.activeDataLake?.bucket ?? 'not connected' }}</div>
						</div>
					</div>
				</div>
				<div class="bg-card border border-border rounded-lg p-4">
					<div class="flex items-center gap-3">
						<Brain class="w-5 h-5 text-violet-500" />
						<div>
							<div class="text-sm font-bold text-foreground">Federated Training</div>
							<div class="text-[10px] font-mono text-foreground/50">{{ pipeline.federatedConfig.status }} / {{ pipeline.federatedConfig.protocol }}</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</section>
</template>
