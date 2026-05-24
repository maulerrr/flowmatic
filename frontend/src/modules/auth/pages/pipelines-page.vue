<script setup lang="ts">
import { type PipelineRun, apiClient } from '@/api/client';
import { AlertCircle, ArrowDownRight, ArrowRight, ArrowUpRight, CheckCircle, CheckSquare, Clock, Database, Download, Eye, Filter, Lightbulb, RefreshCw, Search, Send, Sparkles, ToggleLeft, Trash2, TrendingUp, Upload, X } from 'lucide-vue-next';
import { computed, onMounted, ref } from 'vue';
import { useRoute } from 'vue-router';



import PaginationBar from '@/core/components/pagination-bar/pagination-bar.vue';
import PipelineRunCard from '@/core/components/pipeline-run-card.vue';





const route = useRoute()
const runs = ref<PipelineRun[]>([])
const loading = ref(false)
const selectedRun = ref<PipelineRun | null>(null)
const showPreview = ref(false)
const statusFilter = ref<string>('')
const deleting = ref(false)
const showDeleteConfirm = ref(false)
const runToDelete = ref<PipelineRun | null>(null)
const openMenuId = ref<string | null>(null)
const showExportModal = ref(false)
const exportingRunId = ref<string | null>(null)
const previewData = ref<any>(null)
const previewLoading = ref(false)
const previewPage = ref(1)
const previewPageSize = ref(25)
const exportAdapters = ref<any[]>([])
const selectedAdapter = ref<string>('')
const exportSettings = ref<Record<string, any>>({})
const saveCredentials = ref(false)
const exporting = ref(false)
const feedback = ref<{ type: 'success' | 'error' | 'info'; message: string } | null>(null)

const statusConfig = {
	queued: { color: 'text-warning', bg: 'bg-warning/10', icon: Clock, label: 'Queued' },
	processing: { color: 'text-primary', bg: 'bg-primary/10', icon: RefreshCw, label: 'Processing' },
	completed: { color: 'text-success', bg: 'bg-success/10', icon: CheckCircle, label: 'Completed' },
	failed: { color: 'text-destructive', bg: 'bg-destructive/10', icon: AlertCircle, label: 'Failed' },
}

const sortedRuns = computed(() => {
	let filtered = runs.value
	if (statusFilter.value) {
		filtered = filtered.filter(r => r.status === statusFilter.value)
	}
	return [...filtered].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
})

const runMetrics = computed(() => {
	const total = runs.value.length
	const completed = runs.value.filter(r => r.status === 'completed').length
	const failed = runs.value.filter(r => r.status === 'failed').length
	const inProgress = runs.value.filter(r => r.status === 'processing' || r.status === 'queued').length
	return { total, completed, failed, inProgress }
})

const selectedRunSummary = computed(() => {
	if (!selectedRun.value?.summary) return null
	try {
		const parsed = JSON.parse(selectedRun.value.summary)
		if (parsed.scores && parsed.overview) {
			return parsed
		}
		return {
			overview: selectedRun.value.summary,
			scores: null,
			insights: [],
			recommendation: null
		}
	} catch (e) {
		return {
			overview: selectedRun.value.summary,
			scores: null,
			insights: [],
			recommendation: null
		}
	}
})

const setFeedback = (type: 'success' | 'error' | 'info', message: string) => {
	feedback.value = { type, message }
	setTimeout(() => {
		if (feedback.value === null) return
		feedback.value = null
	}, 5000)
}

const fetchRuns = async () => {
	loading.value = true
	try {
		const response = await apiClient.listPipelineRuns(50, 0, statusFilter.value || undefined)
		if (response.success && response.data) {
			runs.value = response.data
		}
	} catch (error) {
		console.error('Failed to fetch runs:', error)
	} finally {
		loading.value = false
	}
}

const formatDate = (dateString: string) => {
	const date = new Date(dateString)
	return date.toLocaleDateString('en-US', {
		month: 'short',
		day: 'numeric',
		year: 'numeric',
		hour: '2-digit',
		minute: '2-digit',
	})
}

const formatBytes = (bytes: number) => {
	if (bytes === 0) return '0 B'
	const k = 1024
	const sizes = ['B', 'KB', 'MB', 'GB']
	const i = Math.floor(Math.log(bytes) / Math.log(k))
	return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

const getStatusConfig = (status: string) => {
	return statusConfig[status as keyof typeof statusConfig] || statusConfig.queued
}

const selectRun = (run: PipelineRun) => {
	selectedRun.value = run
	showPreview.value = false
}

const showRunPreview = async (run: PipelineRun) => {
	selectRun(run)
	showPreview.value = true
}

const confirmDelete = (run: PipelineRun) => {
	runToDelete.value = run
	showDeleteConfirm.value = true
}

const deleteRun = async () => {
	if (!runToDelete.value) return

	deleting.value = true
	try {
		const response = await apiClient.deleteRun(runToDelete.value.id)
		if (response.success) {
			const deletedId = runToDelete.value!.id
			runs.value = runs.value.filter(r => r.id !== deletedId)
			showDeleteConfirm.value = false
			runToDelete.value = null
			if (selectedRun.value?.id === deletedId) {
				selectedRun.value = null
			}
			setFeedback('success', 'Run deleted successfully')
		}
	} catch (error) {
		console.error('Failed to delete run:', error)
		setFeedback('error', 'Failed to delete run')
	} finally {
		deleting.value = false
	}
}

const showDataPreview = async (run: PipelineRun) => {
	previewLoading.value = true
	selectedRun.value = run
	previewPage.value = 1

	await fetchPreviewPage(run.id, 1)
}

const fetchPreviewPage = async (runId: string, page: number) => {
	try {
		previewLoading.value = true
		const response = await apiClient.previewPipelineData(runId, page, previewPageSize.value)

		if (response.success && response.data) {
			previewData.value = response.data
			previewPage.value = page
			showPreview.value = true
		}
	} catch (error) {
		console.error('Failed to load preview:', error)
		setFeedback('error', 'Failed to load data preview')
	} finally {
		previewLoading.value = false
	}
}

const handlePreviewPageChange = (page: number) => {
	if (selectedRun.value) {
		fetchPreviewPage(selectedRun.value.id, page)
	}
}

const openExportModal = async (run: PipelineRun) => {
	exportingRunId.value = run.id
	showExportModal.value = true
	selectedAdapter.value = ''
	exportSettings.value = { replaceTable: false, ifExists: 'append' }
	saveCredentials.value = false
	openMenuId.value = null

	try {
		const response = await apiClient.getExportAdapters()
		if (response.success && response.data) {
			exportAdapters.value = response.data
		}
	} catch (error) {
		console.error('Failed to load adapters:', error)
	}
}

const performExport = async () => {
	if (!exportingRunId.value || !selectedAdapter.value) {
		setFeedback('error', 'Please select an export destination')
		return
	}

	exporting.value = true
	try {
		// Normalize settings per adapter (e.g., postgres replace vs append)
		const settings = { ...exportSettings.value }
		if (selectedAdapter.value === 'postgres') {
			settings.ifExists = settings.replaceTable ? 'replace' : 'append'
			delete settings.replaceTable
		}
		const response = await apiClient.exportPipelineRun(
			exportingRunId.value,
			selectedAdapter.value,
			settings,
			saveCredentials.value,
		)
		if (response.success) {
			const destination = response.data?.destination || 'Export completed'
			setFeedback('success', response.data?.message || destination)
			showExportModal.value = false
		}
	} catch (error) {
		console.error('Export failed:', error)
	setFeedback('error', `Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
	} finally {
		exporting.value = false
	}
}

const cleanupOldRuns = async () => {
	deleting.value = true
	try {
		const response = await apiClient.cleanupOldRuns(30, ['failed', 'completed'])
		if (response.success) {
			setFeedback('success', `Deleted ${response.data?.count || 0} old runs`)
			await fetchRuns()
		}
	} catch (error) {
		console.error('Failed to cleanup runs:', error)
		setFeedback('error', 'Failed to cleanup old runs')
	} finally {
		deleting.value = false
	}
}

onMounted(() => {
	fetchRuns()

	// If redirected from upload with a runId, select that run
	const runId = route.query.runId as string
	if (runId) {
		setTimeout(() => {
			const run = runs.value.find((r) => r.id === runId)
			if (run) {
				selectRun(run)
			}
		}, 500)
	}

	// Polling for updates every 5 seconds
	const interval = setInterval(fetchRuns, 5000)
	return () => clearInterval(interval)
})
</script>

<template>
	<div class="min-h-screen">
		<section class="max-w-7xl mx-auto px-4 lg:px-8 py-10 space-y-8">
			<!-- Hero & Actions -->
			<div class="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
				<div class="space-y-2">
					<div
						class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/15 text-primary text-xs font-semibold uppercase tracking-[0.18em]"
					>
						<span class="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
						Live pipelines
					</div>
					<h2 class="text-4xl font-bold text-foreground leading-tight">Pipeline Runs</h2>
					<p class="text-base text-foreground/70 max-w-2xl">
						Monitor ingestion, quality checks, and exports with a calmer surface and immediate
						feedback.
					</p>
				</div>
				<div class="flex flex-wrap gap-3">
					<button
						@click="fetchRuns"
						class="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-secondary/10 text-secondary border border-secondary/20 hover:bg-secondary/20 hover:shadow-[var(--glow)] transition"
					>
						<RefreshCw
							class="w-5 h-5"
							:class="{ 'animate-spin': loading }"
						/>
						<span class="font-semibold">Refresh</span>
					</button>
					<button
						@click="cleanupOldRuns"
						:disabled="deleting"
						class="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-destructive/10 text-destructive border border-destructive/20 hover:bg-destructive/20 transition disabled:opacity-50"
					>
						<Trash2 class="w-5 h-5" />
						<span class="font-semibold">Cleanup</span>
					</button>
					<router-link
						to="/upload"
						class="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[var(--gradient-accent)] text-sidebar-foreground font-semibold shadow-[var(--glow)] hover:translate-y-[-1px] transition"
					>
						<Upload class="w-5 h-5" />
						New Upload
					</router-link>
				</div>
			</div>

			<!-- Metrics Strip -->
			<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
				<div
					class="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 shadow-sm hover:border-primary/30 transition-colors"
				>
					<p class="text-xs uppercase tracking-[0.16em] text-foreground/60">Total runs</p>
					<div class="mt-2 flex items-end gap-2">
						<p class="text-3xl font-bold">{{ runMetrics.total }}</p>
						<span class="text-sm text-foreground/60">records</span>
					</div>
				</div>
				<div
					class="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 shadow-sm hover:border-success/30 transition-colors"
				>
					<p class="text-xs uppercase tracking-[0.16em] text-success/80">Completed</p>
					<div class="mt-2 flex items-center gap-2">
						<p class="text-3xl font-bold text-success">{{ runMetrics.completed }}</p>
						<span class="text-sm text-foreground/60">cleaned</span>
					</div>
				</div>
				<div
					class="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 shadow-sm hover:border-destructive/30 transition-colors"
				>
					<p class="text-xs uppercase tracking-[0.16em] text-destructive/80">Failed</p>
					<div class="mt-2 flex items-center gap-2">
						<p class="text-3xl font-bold text-destructive">{{ runMetrics.failed }}</p>
						<span class="text-sm text-foreground/60">need attention</span>
					</div>
				</div>
				<div
					class="rounded-xl border border-border bg-card/50 backdrop-blur-sm p-4 shadow-sm hover:border-primary/30 transition-colors"
				>
					<p class="text-xs uppercase tracking-[0.16em] text-primary">In progress</p>
					<div class="mt-2 flex items-center gap-2">
						<p class="text-3xl font-bold text-primary">{{ runMetrics.inProgress }}</p>
						<span class="text-sm text-foreground/60">running</span>
					</div>
				</div>
			</div>

			<!-- Feedback Banner -->
			<div
				v-if="feedback"
				:class="[
				'rounded-xl border px-4 py-3 flex items-center gap-3 shadow-sm',
				feedback.type === 'success' ? 'bg-success/10 border-success/30 text-success' : '',
				feedback.type === 'error' ? 'bg-destructive/10 border-destructive/30 text-destructive' : '',
				feedback.type === 'info' ? 'bg-primary/10 border-primary/30 text-primary' : ''
			]"
			>
				<AlertCircle
					v-if="feedback.type === 'error'"
					class="w-5 h-5"
				/>
				<CheckCircle
					v-else-if="feedback.type === 'success'"
					class="w-5 h-5"
				/>
				<Clock
					v-else
					class="w-5 h-5"
				/>
				<span class="font-semibold">{{ feedback.message }}</span>
			</div>

			<!-- Filters -->
			<div class="flex flex-wrap gap-2">
				<button
					@click="statusFilter = ''"
					:class="[
						'px-4 py-2 rounded-xl font-semibold text-sm flex items-center gap-2 border transition',
						!statusFilter ? 'bg-primary/15 text-primary border-primary/40 shadow-[var(--glow)]' : 'bg-card border-border text-foreground/70 hover:border-primary/30 hover:text-foreground'
					]"
				>
					<Filter class="w-4 h-4" />
					All
				</button>
				<button
					v-for="status in ['queued', 'processing', 'completed', 'failed']"
					:key="status"
					@click="statusFilter = statusFilter === status ? '' : status"
					:class="[
						'px-4 py-2 rounded-xl font-semibold text-sm capitalize border transition',
						statusFilter === status
							? getStatusConfig(status).bg + ' ' + getStatusConfig(status).color + ' border-primary/40 shadow-[var(--glow)]'
							: 'bg-card border-border text-foreground/70 hover:border-primary/30 hover:text-foreground'
					]"
				>
					{{ getStatusConfig(status).label }}
				</button>
			</div>

			<!-- Loading State -->
			<div
				v-if="loading && runs.length === 0"
				class="space-y-4"
			>
				<div
					v-for="i in 3"
					:key="i"
					class="bg-surface-2 rounded-xl border border-border p-6 animate-pulse"
				>
					<div class="h-6 bg-border rounded w-1/4 mb-4" />
					<div class="h-4 bg-border rounded w-1/2" />
				</div>
			</div>

			<!-- Empty State -->
			<div
				v-else-if="sortedRuns.length === 0"
				class="bg-surface-2 rounded-2xl border border-border p-12 text-center shadow-lg"
			>
				<Database class="w-16 h-16 text-foreground/40 mx-auto mb-4" />
				<h3 class="text-xl font-semibold text-foreground mb-2">No Pipeline Runs Yet</h3>
				<p class="text-foreground/60 mb-6">
					Upload a CSV or JSON file to get started with your first pipeline.
				</p>
				<router-link
					to="/upload"
					class="inline-block px-6 py-3 rounded-xl bg-[var(--gradient-accent)] text-sidebar-foreground font-semibold shadow-[var(--glow)] hover:translate-y-[-1px] transition"
				>
					Upload Data
				</router-link>
			</div>

			<!-- Runs Grid -->
			<div
				v-else
				class="grid gap-6 md:grid-cols-2 xl:grid-cols-3"
			>
				<PipelineRunCard
					v-for="run in sortedRuns"
					:key="run.id"
					:run="run"
					:menu-open="openMenuId === run.id"
					@menu-toggle="openMenuId = openMenuId === run.id ? null : run.id"
					@preview="showDataPreview(run)"
					@export="openExportModal(run)"
					@delete="confirmDelete(run)"
				/>
			</div>
		</section>

		<!-- Run Details Modal -->
		<div
			v-if="selectedRun"
			class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
		>
			<div
				class="bg-card/95 backdrop-blur-xl rounded-2xl border border-border max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
			>
				<div
					class="sticky top-0 bg-card/95 backdrop-blur-xl border-b border-border px-6 py-4 flex items-center justify-between z-10"
				>
					<h2 class="text-2xl font-bold text-foreground">Run Details</h2>
					<button
						@click="selectedRun = null"
						class="text-foreground/60 hover:text-foreground transition rounded-full p-1 hover:bg-foreground/5"
					>
						<X class="w-6 h-6" />
					</button>
				</div>

				<div class="p-6 space-y-6">
					<div>
						<h3 class="text-lg font-semibold text-foreground mb-4">File Information</h3>
						<div class="grid grid-cols-2 gap-4 bg-surface-3 p-4 rounded-xl border border-border">
							<div>
								<p class="text-sm text-foreground/60">File Name</p>
								<p class="text-foreground font-medium">{{ selectedRun.sourceFileName }}</p>
							</div>
							<div>
								<p class="text-sm text-foreground/60">File Size</p>
								<p class="text-foreground font-medium">
									{{ formatBytes(selectedRun.sourceFile?.fileSize || 0) }}
								</p>
							</div>
							<div>
								<p class="text-sm text-foreground/60">Status</p>
								<span
									:class="[
										'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold mt-1',
										getStatusConfig(selectedRun.status).bg,
										getStatusConfig(selectedRun.status).color,
									]"
								>
									<component
										:is="getStatusConfig(selectedRun.status).icon"
										class="w-4 h-4"
									/>
									{{ getStatusConfig(selectedRun.status).label }}
								</span>
							</div>
							<div>
								<p class="text-sm text-foreground/60">Processing Time</p>
								<p class="text-foreground font-medium">{{ selectedRun.processingTimeMs }}ms</p>
							</div>
						</div>
					</div>

					<div>
						<h3 class="text-lg font-semibold text-foreground mb-4">Processing Statistics</h3>
						<div class="grid grid-cols-3 gap-4">
							<div class="bg-primary/10 border border-primary/30 rounded-xl p-4">
								<p class="text-sm text-primary mb-2">Rows Ingested</p>
								<p class="text-3xl font-bold text-foreground">{{ selectedRun.rowsIngested }}</p>
							</div>
							<div class="bg-success/10 border border-success/30 rounded-xl p-4">
								<p class="text-sm text-success mb-2">Rows Cleaned</p>
								<p class="text-3xl font-bold text-success">{{ selectedRun.rowsCleaned }}</p>
							</div>
							<div class="bg-warning/10 border border-warning/30 rounded-xl p-4">
								<p class="text-sm text-warning mb-2">Errors Found</p>
								<p class="text-3xl font-bold text-warning">{{ selectedRun.rowsErrors }}</p>
							</div>
						</div>
					</div>

					<!-- AI Analysis Section -->
					<div
						v-if="selectedRunSummary"
						class="bg-gradient-to-br from-primary/5 to-secondary/5 rounded-xl border border-primary/10 overflow-hidden"
					>
						<!-- Header -->
						<div class="px-4 py-3 border-b border-primary/10 flex items-center gap-2 bg-primary/5">
							<Sparkles class="w-4 h-4 text-primary animate-pulse" />
							<span class="text-sm font-bold text-primary tracking-wide uppercase"
								>AI Analysis</span
							>
						</div>

						<div class="p-4 space-y-5">
							<!-- Overview -->
							<div class="text-sm text-foreground/80 leading-relaxed">
								{{ selectedRunSummary.overview.replace(/\*\*/g, '') }}
							</div>

							<!-- Scores -->
							<div
								v-if="selectedRunSummary.scores"
								class="flex items-center gap-4 bg-white/5 rounded-lg p-3"
							>
								<div class="flex-1 text-center border-r border-white/10">
									<p class="text-xs text-foreground/50 uppercase mb-1">Raw Quality</p>
									<p
										:class="['text-xl font-bold', selectedRunSummary.scores.initial > 80 ? 'text-success' : 'text-warning']"
									>
										{{ selectedRunSummary.scores.initial
										}}<span class="text-xs text-foreground/40 ml-0.5">/100</span>
									</p>
								</div>
								<div class="flex items-center text-foreground/40">
									<ArrowRight class="w-4 h-4" />
								</div>
								<div class="flex-1 text-center">
									<p class="text-xs text-foreground/50 uppercase mb-1">Cleaned Quality</p>
									<p
										:class="['text-xl font-bold', selectedRunSummary.scores.final > 90 ? 'text-primary' : 'text-success']"
									>
										{{ selectedRunSummary.scores.final
										}}<span class="text-xs text-foreground/40 ml-0.5">/100</span>
									</p>
								</div>
							</div>

							<!-- Key Insights -->
							<div v-if="selectedRunSummary.insights?.length">
								<h4
									class="text-xs font-semibold text-foreground/70 uppercase tracking-wider mb-2 flex items-center gap-2"
								>
									<Lightbulb class="w-3.5 h-3.5" /> Key Insights
								</h4>
								<ul class="space-y-2">
									<li
										v-for="(insight, idx) in selectedRunSummary.insights"
										:key="idx"
										class="flex gap-2 text-sm text-foreground/70"
									>
										<span
											class="block w-1.5 h-1.5 mt-1.5 rounded-full bg-primary/50 flex-shrink-0"
										/>
										{{ insight }}
									</li>
								</ul>
							</div>

							<!-- Actions -->
							<div
								v-if="selectedRunSummary.recommendation"
								class="bg-primary/10 rounded-lg p-3 flex gap-3 items-start border border-primary/20"
							>
								<CheckSquare class="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
								<div>
									<p class="text-xs font-bold text-primary uppercase mb-0.5">Recommended Action</p>
									<p class="text-sm text-foreground/90">{{ selectedRunSummary.recommendation }}</p>
								</div>
							</div>
						</div>
					</div>

					<div
						v-if="selectedRun.errorMessage"
						class="bg-destructive/10 border border-destructive/30 rounded-xl p-4"
					>
						<h3 class="text-sm font-semibold text-destructive mb-2">Error Details</h3>
						<p class="text-destructive/80 text-sm">{{ selectedRun.errorMessage }}</p>
					</div>

					<div class="pt-4 border-t border-white/10 flex gap-2 justify-end">
						<button
							@click="confirmDelete(selectedRun)"
							class="px-4 py-2 rounded-xl bg-destructive text-sidebar-foreground font-semibold hover:bg-destructive/90 shadow-lg transition flex items-center gap-2"
						>
							<Trash2 class="w-4 h-4" />
							Delete Run
						</button>
						<button
							@click="selectedRun = null"
							class="px-4 py-2 rounded-xl border border-white/10 hover:bg-white/5 text-foreground font-semibold transition"
						>
							Close
						</button>
					</div>
				</div>
			</div>
		</div>

		<!-- Delete Confirmation Modal -->
		<div
			v-if="showDeleteConfirm && runToDelete"
			class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
		>
			<div
				class="bg-card/95 backdrop-blur-xl rounded-2xl border border-white/10 max-w-md w-full p-6 shadow-2xl transform transition-all scale-100"
			>
				<div
					class="flex items-center justify-center w-12 h-12 rounded-full bg-destructive/10 border border-destructive/30 mx-auto mb-4 animate-pulse"
				>
					<AlertCircle class="w-6 h-6 text-destructive" />
				</div>
				<h2 class="text-xl font-bold text-foreground text-center mb-2">Delete Run?</h2>
				<p class="text-foreground/70 text-center mb-6">
					This will permanently delete "{{ runToDelete.sourceFileName }}" and all associated data
					from S3. This action cannot be undone.
				</p>
				<div class="flex gap-3 justify-center">
					<button
						@click="showDeleteConfirm = false; runToDelete = null"
						class="px-4 py-2 rounded-xl border border-white/10 hover:bg-white/5 text-foreground font-semibold transition"
					>
						Cancel
					</button>
					<button
						@click="deleteRun"
						:disabled="deleting"
						class="px-4 py-2 rounded-xl bg-destructive text-sidebar-foreground font-semibold hover:bg-destructive/90 shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
					>
						{{ deleting ? 'Deleting...' : 'Delete' }}
					</button>
				</div>
			</div>
		</div>

		<!-- Preview Data Modal -->
		<div
			v-if="showPreview && previewData"
			class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
		>
			<div
				class="bg-card/95 backdrop-blur-xl rounded-2xl border border-white/10 max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl"
			>
				<div
					class="sticky top-0 bg-card/95 backdrop-blur-xl border-b border-white/10 px-6 py-4 flex items-center justify-between z-10"
				>
					<div>
						<h2 class="text-2xl font-bold text-foreground">Data Preview</h2>
						<p
							class="text-sm text-foreground/60"
							v-if="previewData?.meta"
						>
							{{ previewData.meta.fileName }} • Page {{ previewData.pagination?.page }} of
							{{ previewData.pagination?.totalPages }} ({{ previewData.pagination?.totalCount }}
							rows)
						</p>
					</div>
					<button
						@click="showPreview = false; previewData = null"
						class="text-foreground/60 hover:text-foreground transition rounded-full p-1 hover:bg-white/5"
					>
						<X class="w-6 h-6" />
					</button>
				</div>

				<div class="p-6">
					<div
						v-if="previewData?.meta?.columns"
						class="overflow-x-auto rounded-lg border border-white/10"
					>
						<table class="w-full text-sm">
							<thead class="border-b border-white/10 bg-white/5">
								<tr>
									<th
										v-for="col in previewData.meta.columns"
										:key="col"
										class="text-left px-4 py-3 text-foreground/80 font-semibold uppercase tracking-[0.08em] whitespace-nowrap"
									>
										{{ col }}
									</th>
								</tr>
							</thead>
							<tbody class="relative">
								<div
									v-if="previewLoading"
									class="absolute inset-0 bg-background/50 flex items-center justify-center z-10 min-h-[200px]"
								>
									<RefreshCw class="animate-spin w-8 h-8 text-primary" />
								</div>
								<tr
									v-for="(row, idx) in previewData.data"
									:key="idx"
									class="border-b border-white/5 hover:bg-white/5 transition-colors last:border-0"
								>
									<td
										v-for="col in previewData.meta.columns"
										:key="col"
										class="px-4 py-3 text-foreground/70 whitespace-nowrap"
									>
										{{ row[col] }}
									</td>
								</tr>
							</tbody>
						</table>
					</div>

					<!-- Pagination Controls -->
					<div
						class="mt-4"
						v-if="previewData?.pagination && previewData.pagination.totalPages > 1"
					>
						<PaginationBar
							:total-pages="previewData.pagination.totalPages"
							:total-count="previewData.pagination.totalCount"
							:page-size="previewData.pagination.pageSize"
							:default-page="previewPage"
							@page-change="handlePreviewPageChange"
						/>
					</div>
					<div
						class="mt-4 p-4 bg-white/5 rounded-xl border border-white/10 flex items-center justify-between"
					>
						<p class="text-sm text-foreground/70">
							Total rows in dataset:
							<span
								class="text-foreground font-semibold"
								>{{ previewData.pagination?.totalCount || 0 }}</span
							>
						</p>
					</div>
				</div>

				<div class="border-t border-white/10 px-6 py-4 flex gap-2 justify-end">
					<button
						@click="showPreview = false; previewData = null"
						class="px-4 py-2 rounded-xl border border-white/10 hover:bg-white/5 text-foreground font-semibold transition"
					>
						Close
					</button>
				</div>
			</div>
		</div>

		<!-- Export Modal -->
		<div
			v-if="showExportModal && exportingRunId"
			class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
		>
			<div
				class="bg-card/95 backdrop-blur-xl rounded-2xl border border-white/10 max-w-2xl w-full shadow-2xl"
			>
				<div class="border-b border-white/10 px-6 py-4 flex items-center justify-between">
					<h2
						class="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent"
					>
						Export Data
					</h2>
					<button
						@click="showExportModal = false"
						class="text-foreground/60 hover:text-foreground transition rounded-full p-1 hover:bg-white/5"
					>
						<X class="w-6 h-6" />
					</button>
				</div>

				<div class="p-6 space-y-6">
					<div>
						<label class="block text-sm font-semibold text-foreground mb-3"
							>Export Destination</label
						>
						<div class="grid grid-cols-2 gap-3">
							<button
								v-for="adapter in exportAdapters"
								:key="adapter.type"
								@click="selectedAdapter = adapter.type"
								:class="[
									'p-4 rounded-xl border-2 transition text-left',
									selectedAdapter === adapter.type
										? 'border-primary bg-primary/10 shadow-[var(--glow)]'
										: 'border-border bg-surface-3 hover:border-primary/30'
								]"
							>
								<p class="font-semibold text-foreground">{{ adapter.name }}</p>
								<p class="text-xs text-foreground/60 mt-1">{{ adapter.description }}</p>
							</button>
						</div>
					</div>

					<!-- Dynamic Settings Based on Adapter -->
					<div
						v-if="selectedAdapter"
						class="space-y-4"
					>
						<div v-if="selectedAdapter === 'postgres'">
							<label class="block text-sm font-semibold text-foreground mb-2"
								>PostgreSQL Settings</label
							>
							<input
								v-model="exportSettings.host"
								placeholder="Host"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.port"
								placeholder="Port"
								type="number"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.database"
								placeholder="Database"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.table"
								placeholder="Table Name"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.username"
								placeholder="Username"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.password"
								placeholder="Password"
								type="password"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground"
							/>
							<div class="mt-3 flex items-center gap-2">
								<input
									id="replace-table"
									type="checkbox"
									v-model="exportSettings.replaceTable"
									class="w-4 h-4 text-primary bg-surface-1 border-border rounded"
								/>
								<label
									for="replace-table"
									class="text-sm text-foreground/80"
									>Replace existing table (drop & recreate to match CSV columns)</label
								>
							</div>
						</div>

						<div v-if="selectedAdapter === 'mongodb'">
							<label class="block text-sm font-semibold text-foreground mb-2"
								>MongoDB Settings</label
							>
							<input
								v-model="exportSettings.uri"
								placeholder="MongoDB URI (mongodb://...)"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.database"
								placeholder="Database Name"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.collection"
								placeholder="Collection Name"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground"
							/>
						</div>

						<div v-if="selectedAdapter === 'huggingface'">
							<label class="block text-sm font-semibold text-foreground mb-2"
								>Hugging Face Settings</label
							>
							<input
								v-model="exportSettings.token"
								placeholder="Hugging Face API Token (leave empty to use saved token)"
								type="password"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.repoName"
								placeholder="Repository Name"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground mb-2"
							/>
							<input
								v-model="exportSettings.fileName"
								placeholder="File Name (default: cleaned_data.csv)"
								class="w-full px-3 py-2 bg-surface-1 border border-border rounded-lg text-foreground"
							/>
						</div>

						<label
							v-if="['postgres', 'mongodb', 'huggingface'].includes(selectedAdapter)"
							class="flex items-center gap-2 text-foreground/80 rounded-xl border border-border p-3 bg-surface-3"
						>
							<input
								v-model="saveCredentials"
								type="checkbox"
								class="rounded"
							/>
							Save credentials encrypted for this organization
						</label>

						<div v-if="selectedAdapter === 'csv'">
							<label class="block text-sm font-semibold text-foreground mb-2">CSV Settings</label>
							<label class="flex items-center gap-2 text-foreground/80">
								<input
									v-model="exportSettings.includeIndex"
									type="checkbox"
									class="rounded"
								/>
								Include Index Column
							</label>
						</div>

						<div v-if="selectedAdapter === 'json'">
							<label class="block text-sm font-semibold text-foreground mb-2">JSON Settings</label>
							<label class="flex items-center gap-2 text-foreground/80">
								<input
									v-model="exportSettings.prettyPrint"
									type="checkbox"
									class="rounded"
								/>
								Pretty Print
							</label>
						</div>
					</div>
				</div>

				<div class="border-t border-white/10 px-6 py-4 flex gap-2 justify-end bg-white/5">
					<button
						@click="showExportModal = false"
						class="px-4 py-2 rounded-xl border border-white/10 hover:bg-white/5 text-foreground font-semibold transition"
					>
						Cancel
					</button>
					<button
						@click="performExport"
						:disabled="exporting || !selectedAdapter"
						class="px-4 py-2 rounded-xl bg-gradient-to-r from-primary to-secondary text-white font-semibold shadow-lg hover:shadow-primary/25 hover:scale-[1.02] transition disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-2"
					>
						<Send class="w-4 h-4" />
						{{ exporting ? 'Exporting...' : 'Export' }}
					</button>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped></style>
