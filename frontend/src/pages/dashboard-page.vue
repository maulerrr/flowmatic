<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { TrendingUp, FileText, CheckCircle, AlertCircle, Clock, ArrowUpRight, ArrowDownRight, Target } from 'lucide-vue-next'
import { apiClient } from '@/api/client'

const stats = ref([
	{
		icon: FileText,
		label: 'Total Uploads',
		value: '0',
		change: '-',
		changePercent: 0,
		positive: null,
		color: 'primary',
		bg: 'bg-primary/10',
	},
	{
		icon: CheckCircle,
		label: 'Completed Runs',
		value: '0',
		change: '-',
		changePercent: 0,
		positive: true,
		color: 'success',
		bg: 'bg-success/10',
	},
	{
		icon: Clock,
		label: 'In Progress',
		value: '0',
		change: '-',
		changePercent: 0,
		positive: null,
		color: 'warning',
		bg: 'bg-warning/10',
	},
	{
		icon: AlertCircle,
		label: 'Failed Runs',
		value: '0',
		change: '-',
		changePercent: 0,
		positive: true,
		color: 'destructive',
		bg: 'bg-destructive/10',
	},
])

const recentActivity = ref<any[]>([])
const loading = ref(false)

const getColorClass = (color: string) => {
	const colors: Record<string, string> = {
		primary: 'text-primary',
		success: 'text-success',
		warning: 'text-warning',
		destructive: 'text-destructive',
	}
	return colors[color] || 'text-primary'
}

const getChangeIcon = (positive: boolean | null) => {
	if (positive === true) return ArrowUpRight
	if (positive === false) return ArrowDownRight
	return null
}

const fetchData = async () => {
	loading.value = true
	try {
		const [sumRes, runRes] = await Promise.all([apiClient.getAnalyticsSummary(), apiClient.listPipelineRuns(5)])

		if (sumRes.success && sumRes.data) {
			const s = sumRes.data
			stats.value[0].value = s.total.toString()
			stats.value[1].value = s.completed.toString()
			stats.value[2].value = s.inProgress.toString()
			stats.value[3].value = s.failed.toString()
		}

		if (runRes.success && runRes.data) {
			recentActivity.value = runRes.data.map((r: any) => ({
				id: r.id,
				title: r.sourceFileName,
				status: r.status,
				progress: r.status === 'completed' ? 100 : r.status === 'failed' ? 100 : 50, // simple mock for progress
				time: new Date(r.createdAt).toLocaleString(),
				size: r.sourceFile?.fileSize ? (r.sourceFile.fileSize / (1024 * 1024)).toFixed(2) + ' MB' : 'N/A',
			}))
		}
	} catch (error) {
		console.error('Failed to fetch dashboard data', error)
	} finally {
		loading.value = false
	}
}

onMounted(() => {
	fetchData()
})
</script>

<template>
	<div class="min-h-screen bg-background">
		<!-- Header Section -->
		<div class="px-6 py-8 md:px-8 border-b border-border/50">
			<div class="max-w-7xl mx-auto">
				<div>
					<h1 class="text-3xl md:text-4xl font-bold text-foreground mb-2">Dashboard</h1>
					<p class="text-foreground/60">Monitor your data pipelines and processing metrics</p>
				</div>
			</div>
		</div>

		<!-- Main Content -->
		<div class="px-6 py-8 md:px-8 max-w-7xl mx-auto">
			<!-- Stats Grid -->
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
				<div 
					v-for="stat in stats" 
					:key="stat.label"
					class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-6 hover:border-primary/40 transition-all duration-300 group hover:shadow-[var(--glow)]"
				>
					<div class="flex items-start justify-between mb-4">
						<div :class="[stat.bg, 'w-12 h-12 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform']">
							<component :is="stat.icon" :class="['w-6 h-6', getColorClass(stat.color)]" />
						</div>
					</div>
					<p class="text-foreground/70 text-sm font-medium mb-1">{{ stat.label }}</p>
					<div class="flex items-end justify-between">
						<p class="text-3xl font-bold text-foreground">{{ stat.value }}</p>
					</div>
				</div>
			</div>

			<!-- Main Grid -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Recent Activity - Takes 2 columns -->
				<div class="lg:col-span-2">
					<div class="rounded-xl border border-border bg-card/70 backdrop-blur-md overflow-hidden">
						<div class="px-6 py-4 border-b border-border/50 flex items-center justify-between">
							<h3 class="text-xl font-bold text-foreground">Recent Activity</h3>
							<a href="/pipelines" class="text-primary hover:text-primary/80 text-sm font-semibold transition">View all →</a>
						</div>
						
						<div class="divide-y divide-border/30">
							<div 
								v-for="activity in recentActivity" 
								:key="activity.id"
								class="px-6 py-4 hover:bg-card/50 transition-colors group"
							>
								<div class="flex items-start justify-between mb-3">
									<div class="flex-1">
										<h4 class="font-semibold text-foreground group-hover:text-primary transition">{{ activity.title }}</h4>
										<div class="flex items-center gap-4 mt-1">
											<span class="text-xs text-foreground/60">{{ activity.time }}</span>
											<span class="text-xs text-foreground/50">{{ activity.size }}</span>
										</div>
									</div>
									<span
										:class="[
											'px-3 py-1 rounded-full text-xs font-semibold',
											activity.status === 'completed'
												? 'bg-success/20 text-success'
												: 'bg-primary/20 text-primary',
										]"
									>
										{{ activity.status }}
									</span>
								</div>
								
								<!-- Progress Bar -->
								<div class="w-full bg-border/30 rounded-full h-1.5 overflow-hidden">
									<div
										class="h-full rounded-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
										:style="{ width: activity.progress + '%' }"
									/>
								</div>
								<p class="text-xs text-foreground/50 mt-2">{{ activity.progress }}% complete</p>
							</div>
						</div>

						<!-- Footer -->
						<div class="px-6 py-4 border-t border-border/50 bg-card/50 backdrop-blur-md">
							<a href="/upload" class="text-sm text-primary hover:text-primary/80 font-semibold transition">
								Start new upload →
							</a>
						</div>
					</div>
				</div>

				<!-- Right Sidebar -->
				<div class="space-y-6">
					<!-- Quick Actions -->
					<div class="rounded-xl border border-border bg-card/40 backdrop-blur-sm overflow-hidden">
						<div class="px-6 py-4 border-b border-border/50">
							<h3 class="text-lg font-bold text-foreground">Quick Actions</h3>
						</div>
						<div class="p-6 space-y-3">
							<a
								href="/upload"
								class="flex items-center justify-center gap-2 w-full px-4 py-3 bg-gradient-to-r from-primary to-secondary hover:shadow-[var(--glow)] text-foreground rounded-lg font-semibold transition-all duration-200 group"
							>
								<FileText class="w-5 h-5 group-hover:scale-110 transition-transform" />
								<span>New Upload</span>
							</a>
							<button class="w-full px-4 py-3 bg-border/30 hover:bg-border/50 text-foreground rounded-lg font-semibold transition-colors">
								View Reports
							</button>
							<button class="w-full px-4 py-3 bg-border/30 hover:bg-border/50 text-foreground rounded-lg font-semibold transition-colors">
								Schedule Task
							</button>
						</div>
					</div>

					<!-- Info Card -->
					<div class="rounded-xl border border-primary/30 bg-primary/5 backdrop-blur-sm overflow-hidden">
						<div class="p-6 space-y-3">
							<div class="flex items-start gap-3">
								<Target class="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
								<div>
									<h4 class="font-semibold text-foreground text-sm mb-1">Performance Goal</h4>
									<p class="text-xs text-foreground/70">Achieve 25+ completed runs this month</p>
								</div>
							</div>
							<div class="pt-3 border-t border-primary/20">
								<div class="flex items-center justify-between mb-2">
									<span class="text-xs font-semibold text-foreground/80">Progress</span>
									<span class="text-xs font-bold text-primary">72%</span>
								</div>
								<div class="w-full bg-primary/20 rounded-full h-2 overflow-hidden">
									<div class="w-[72%] h-full bg-gradient-to-r from-primary to-secondary rounded-full" />
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped></style>
