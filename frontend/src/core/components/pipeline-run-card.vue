<script setup lang="ts">
import type { PipelineRun } from '@/api/client';
import { AlertCircle, CheckCircle, Clock, Download, Eye, MoreVertical, RefreshCw, Trash2 } from 'lucide-vue-next';





interface Props {
	run: PipelineRun
	menuOpen: boolean
	onMenuToggle: () => void
	onPreview: () => void
	onExport: () => void
	onDelete: () => void
}

withDefaults(defineProps<Props>(), {})

const statusConfig = {
	queued: { color: 'text-warning', bg: 'bg-warning/10', icon: Clock, label: 'Queued' },
	processing: { color: 'text-primary', bg: 'bg-primary/10', icon: RefreshCw, label: 'Processing' },
	completed: { color: 'text-success', bg: 'bg-success/10', icon: CheckCircle, label: 'Completed' },
	failed: { color: 'text-destructive', bg: 'bg-destructive/10', icon: AlertCircle, label: 'Failed' },
}

const getStatusConfig = (status: string) => {
	return statusConfig[status as keyof typeof statusConfig] || statusConfig.queued
}

const formatBytes = (bytes: number) => {
	if (bytes === 0) return '0 B'
	const k = 1024
	const sizes = ['B', 'KB', 'MB', 'GB']
	const i = Math.floor(Math.log(bytes) / Math.log(k))
	return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

const formatDate = (dateString: string) => {
	const date = new Date(dateString)
	return date.toLocaleDateString('en-US', {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit',
	})
}
</script>

<template>
	<div
		class="rounded-xl border border-border bg-card/40 backdrop-blur-sm hover:border-primary/40 transition-all group overflow-hidden"
	>
		<!-- Card Header -->
		<div class="p-6 border-b border-border/30 bg-card/50 flex items-start justify-between">
			<div class="flex-1">
				<h3 class="font-semibold text-foreground mb-1 group-hover:text-primary transition">
					{{ run.sourceFileName }}
				</h3>
				<p class="text-xs text-foreground/60">{{ formatDate(run.createdAt) }}</p>
			</div>
			<div class="flex items-center gap-2">
				<span
					:class="[
						'inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold',
						getStatusConfig(run.status).bg,
						getStatusConfig(run.status).color,
					]"
				>
					<component
						:is="getStatusConfig(run.status).icon"
						class="w-3.5 h-3.5"
					/>
					{{ getStatusConfig(run.status).label }}
				</span>
			</div>
		</div>

		<!-- Card Content -->
		<div class="p-6 space-y-4">
			<!-- File Info -->
			<div class="grid grid-cols-2 gap-4">
				<div>
					<p class="text-xs text-foreground/60 font-medium uppercase tracking-wider">File Size</p>
					<p class="text-sm font-semibold text-foreground mt-1">
						{{ formatBytes(run.sourceFile?.fileSize || 0) }}
					</p>
				</div>
				<div>
					<p class="text-xs text-foreground/60 font-medium uppercase tracking-wider">Records</p>
					<p class="text-sm font-semibold text-foreground mt-1">{{ run.rowsIngested || 0 }}</p>
				</div>
			</div>

			<!-- Progress Bar -->
			<div
				v-if="run.status === 'processing'"
				class="space-y-2"
			>
				<div class="flex items-center justify-between">
					<p class="text-xs text-foreground/60">Progress</p>
					<p class="text-xs font-semibold text-foreground">
						{{ run.rowsCleaned && run.rowsIngested ? Math.round((run.rowsCleaned / run.rowsIngested) * 100) : 0

						}}%
					</p>
				</div>
				<div class="w-full bg-border/30 rounded-full h-2 overflow-hidden">
					<div
						class="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300"
						:style="{ width: (run.rowsCleaned && run.rowsIngested ? Math.round((run.rowsCleaned / run.rowsIngested) * 100) : 0) + '%' }"
					/>
				</div>
			</div>

			<!-- ID -->
			<div class="pt-2 border-t border-border/20">
				<p class="text-xs text-foreground/50 font-mono">ID: {{ run.id.substring(0, 12) }}...</p>
			</div>
		</div>

		<!-- Card Footer - Actions -->
		<div
			class="px-6 py-3 border-t border-border/30 bg-card/30 backdrop-blur-sm flex items-center justify-between"
		>
			<div class="flex items-center gap-1.5">
				<button
					@click="onPreview"
					class="p-2 rounded-lg text-foreground/70 hover:text-foreground hover:bg-border/50 transition group/btn"
					:title="'Preview data'"
				>
					<Eye class="w-4 h-4 group-hover/btn:scale-110 transition-transform" />
				</button>
				<button
					@click="onExport"
					class="p-2 rounded-lg text-foreground/70 hover:text-foreground hover:bg-border/50 transition group/btn"
					:title="'Export data'"
				>
					<Download class="w-4 h-4 group-hover/btn:scale-110 transition-transform" />
				</button>
			</div>
			<button
				@click="onDelete"
				class="p-2 rounded-lg text-destructive/70 hover:text-destructive hover:bg-destructive/10 transition group/btn"
				:title="'Delete run'"
			>
				<Trash2 class="w-4 h-4 group-hover/btn:scale-110 transition-transform" />
			</button>
		</div>
	</div>
</template>

<style scoped></style>
