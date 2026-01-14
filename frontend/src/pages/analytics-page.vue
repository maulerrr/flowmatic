<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { BarChart3, TrendingUp, CalendarDays, Download, RefreshCw, PieChart } from 'lucide-vue-next'
import { apiClient } from '@/api/client'

const timeRanges = [
	{ label: 'Last 7 days', value: '7d' },
	{ label: 'Last 30 days', value: '30d' },
	{ label: 'Last 90 days', value: '90d' },
	{ label: 'All time', value: 'all' },
]

const selectedRange = ref('7d')
const loading = ref(false)
const summary = ref<any>(null)
const chartsData = ref<any>(null)

const metrics = ref([
	{ label: 'Total Data Processed', value: '0 B', change: '-', icon: TrendingUp },
	{ label: 'Avg Processing Time', value: '0s', change: '-', icon: BarChart3 },
	{ label: 'Success Rate', value: '0%', change: '-', icon: PieChart },
	{ label: 'Total Runs', value: '0', change: '-', icon: TrendingUp },
])

const charts = ref([
	{ title: 'Upload Trends', icon: TrendingUp, color: 'from-blue-500 to-cyan-500', class: 'text-blue-500', key: 'uploadTrends', type: 'area' },
	{ title: 'Status Distribution', icon: PieChart, color: 'from-purple-500 to-pink-500', class: 'text-purple-500', key: 'statusDistribution', type: 'donut' },
])

const doughnutColors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6']

const getDonutStyle = (values: number[]) => {
	const total = values.reduce((a, b) => a + b, 0) || 1
	let current = 0
	
	// If no data, show empty ring
	if (values.every(v => v === 0)) {
		return { background: 'conic-gradient(var(--muted) 0% 100%)' }
	}

	const gradientParts = values.map((val, i) => {
		const start = (current / total) * 100
		current += val
		const end = (current / total) * 100
		return `${doughnutColors[i % doughnutColors.length]} ${start}% ${end}%`
	})

	return { background: `conic-gradient(${gradientParts.join(', ')})` }
}

const getPaths = (values: number[]) => {
	if (!values || values.length === 0) return { area: '', line: '' }
	const max = Math.max(...values) || 10
	
	// Create points for the line
	const points = values.map((val, idx) => {
		const x = (idx / (values.length - 1 || 1)) * 100
		const y = 100 - ((val / max) * 90) // Updated to use more height (90%)
		return `${x},${y}`
	})

	// If single point, create a flat line
	if (points.length === 1) {
		points.push(`100,${points[0].split(',')[1]}`)
	}

	const linePath = `M ${points.join(' L ')}`;
	const areaPath = `${linePath} L 100,100 L 0,100 Z`;
	
	return { area: areaPath, line: linePath }
}

const formatBytes = (bytes: number) => {
	if (bytes === 0) return '0 B'
	const k = 1024
	const sizes = ['B', 'KB', 'MB', 'GB']
	const i = Math.floor(Math.log(bytes) / Math.log(k))
	return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatMs = (ms: number) => {
	if (ms < 1000) return ms + 'ms'
	const sec = Math.floor(ms / 1000)
	const min = Math.floor(sec / 60)
	if (min === 0) return sec + 's'
	return `${min}m ${sec % 60}s`
}

const fetchData = async () => {
	loading.value = true
	try {
		const [sumRes, chartRes] = await Promise.all([apiClient.getAnalyticsSummary(), apiClient.getAnalyticsCharts(selectedRange.value)])

		if (sumRes.success && sumRes.data) {
			summary.value = sumRes.data
			metrics.value = [
				{ label: 'Total Data Processed', value: formatBytes(summary.value.totalDataProcessed), change: '-', icon: TrendingUp },
				{ label: 'Avg Processing Time', value: formatMs(summary.value.avgProcessingTime), change: '-', icon: BarChart3 },
				{ label: 'Success Rate', value: summary.value.successRate.toFixed(1) + '%', change: '-', icon: PieChart },
				{ label: 'Total Runs', value: summary.value.total.toString(), change: '-', icon: TrendingUp },
			]
		}

		if (chartRes.success && chartRes.data) {
			chartsData.value = chartRes.data
		}
	} catch (error) {
		console.error('Failed to fetch analytics', error)
	} finally {
		loading.value = false
	}
}

watch(selectedRange, () => {
	fetchData()
})

onMounted(() => {
	fetchData()
})
</script>

<template>
	<div class="min-h-screen bg-background">
		<!-- Header -->
		<div class="px-6 py-8 md:px-8 border-b border-border/50">
			<div class="max-w-7xl mx-auto">
				<h1 class="text-3xl md:text-4xl font-bold text-foreground mb-2">Analytics</h1>
				<p class="text-foreground/60">Monitor platform metrics and pipeline performance</p>
			</div>
		</div>

		<!-- Content -->
		<div class="max-w-7xl mx-auto px-6 py-8 md:px-8">
			<!-- Controls -->
			<div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
				<div class="flex gap-2 flex-wrap">
					<button
						v-for="range in timeRanges"
						:key="range.value"
						@click="selectedRange = range.value"
						:class="[
							'px-4 py-2 rounded-lg border text-sm font-medium transition',
							selectedRange === range.value ? 'bg-primary/10 border-primary text-primary' : 'border-border hover:border-primary/40 text-foreground/80 hover:text-foreground',
						]"
					>
						{{ range.label }}
					</button>
				</div>
				<div class="flex gap-2">
					<button
						@click="fetchData"
						class="flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:border-primary/40 text-foreground/80 hover:text-foreground text-sm font-medium transition"
					>
						<RefreshCw class="w-4 h-4" :class="{ 'animate-spin': loading }" />
						Refresh
					</button>
					<button class="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-primary to-secondary text-foreground font-medium transition hover:shadow-[var(--glow)]">
						<Download class="w-4 h-4" />
						Export
					</button>
				</div>
			</div>

			<!-- Metrics Grid -->
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
				<div v-for="metric in metrics" :key="metric.label" class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-6 hover:border-primary/40 transition-all group">
					<div class="flex items-start justify-between mb-4">
						<div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform">
							<component :is="metric.icon" class="w-5 h-5 text-primary" />
						</div>
					</div>
					<p class="text-foreground/70 text-sm font-medium mb-1">{{ metric.label }}</p>
					<p class="text-3xl font-bold text-foreground">{{ metric.value }}</p>
				</div>
			</div>

			<!-- Charts Grid -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
				<div v-for="chart in charts" :key="chart.title" class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-6 hover:border-primary/40 transition-all">
					<div class="flex items-center gap-4 mb-6">
						<div :class="['w-12 h-12 rounded-lg bg-gradient-to-br flex items-center justify-center', chart.color]">
							<component :is="chart.icon" class="w-6 h-6 text-white" />
						</div>
						<h3 class="text-xl font-bold text-foreground">{{ chart.title }}</h3>
					</div>
					<div class="h-64 flex items-center justify-center bg-card/30 rounded-lg border border-border/30 overflow-hidden relative">
						<div v-if="chartsData && chartsData[chart.key]" class="w-full h-full p-4">
							<!-- Area Chart -->
							<div v-if="chart.type === 'area'" class="w-full h-full relative px-2 pt-4">
								<svg class="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
									<defs>
										<linearGradient :id="'grad-' + chart.key" x1="0" x2="0" y1="0" y2="1">
											<stop offset="0%" stop-color="currentColor" stop-opacity="0.5" :class="chart.class" />
											<stop offset="100%" stop-color="currentColor" stop-opacity="0.05" :class="chart.class" />
										</linearGradient>
									</defs>
									<path :d="getPaths(chartsData[chart.key].values).area" :fill="'url(#grad-' + chart.key + ')'" :class="chart.class" class="transition-all duration-500" />
									<path :d="getPaths(chartsData[chart.key].values).line" fill="none" stroke="currentColor" stroke-width="3" :class="chart.class" class="transition-all duration-500" vector-effect="non-scaling-stroke" />
								</svg>
								
								<div class="absolute inset-0 flex items-stretch mx-2 mt-4">
									<div
										v-for="(val, idx) in chartsData[chart.key].values"
										:key="idx"
										class="flex-1 relative group/bar hover:bg-foreground/5 transition-colors"
									>
										<div class="absolute bottom-4 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-xs px-2 py-1 rounded shadow-lg border border-border opacity-0 group-hover/bar:opacity-100 whitespace-nowrap z-20 pointer-events-none transition-all">
											<span class="font-bold">{{ val }}</span>
											<span class="opacity-70 ml-1">{{ chartsData[chart.key].labels[idx] }}</span>
										</div>
									</div>
								</div>
							</div>

							<!-- Donut Chart -->
							<div v-else-if="chart.type === 'donut'" class="w-full h-full flex items-center justify-center gap-8">
								<div class="relative w-48 h-48 rounded-full shadow-lg border-4 border-card" :style="getDonutStyle(chartsData[chart.key].values)">
									<!-- Center hole -->
									<div class="absolute inset-4 bg-card rounded-full flex items-center justify-center flex-col shadow-inner">
										<span class="text-3xl font-bold tracking-tight">{{ chartsData[chart.key].values.reduce((a: any, b: any) => a + b, 0) }}</span>
										<span class="text-xs text-muted-foreground uppercase font-semibold tracking-wider">Total</span>
									</div>
								</div>
								<!-- Legend -->
								<div class="flex flex-col gap-3 justify-center min-w-[120px]">
									<div v-for="(label, i) in chartsData[chart.key].labels" :key="label" class="flex items-center gap-3">
										<div class="w-3 h-3 rounded-full shadow-sm" :style="{ backgroundColor: doughnutColors[i % doughnutColors.length] }"></div>
										<div class="flex flex-col">
											<span class="text-sm font-medium leading-none">{{ label }}</span>
											<span class="text-xs text-muted-foreground">{{ chartsData[chart.key].values[i] }} ({{ Math.round((chartsData[chart.key].values[i] / (chartsData[chart.key].values.reduce((a: any, b: any) => a + b, 0) || 1)) * 100) }}%)</span>
										</div>
									</div>
								</div>
							</div>

							<!-- Fallback Bar Chart -->
							<div v-else class="w-full h-full flex items-end gap-3 px-2 pb-2">
								<div
									v-for="(val, idx) in chartsData[chart.key].values"
									:key="idx"
									class="flex-1 relative group/bar"
									:style="{ height: `${Math.max((val / (Math.max(...chartsData[chart.key].values) || 1)) * 100, 4)}%` }"
								>
									<div :class="['w-full h-full rounded-t-md opacity-80 group-hover/bar:opacity-100 transition-all duration-300', chart.class.replace('text-', 'bg-')]"></div>
									<div class="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground text-xs px-2 py-1 rounded shadow-lg border border-border opacity-0 group-hover/bar:opacity-100 whitespace-nowrap z-20 pointer-events-none transition-all translate-y-2 group-hover/bar:translate-y-0">
										{{ chartsData[chart.key].labels[idx] }}: {{ val }}
									</div>
								</div>
							</div>
						</div>
						<p v-else-if="loading" class="text-foreground/60 animate-pulse">Loading data...</p>
						<p v-else class="text-foreground/60">No data available</p>
					</div>
				</div>
			</div>

			<!-- Summary Stats -->
			<div v-if="summary" class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-8">
				<h3 class="text-xl font-bold text-foreground mb-6">Summary</h3>
				<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
					<div class="pb-4 border-b border-border/20">
						<p class="text-foreground/70 text-sm font-medium mb-2">Avg. Processing Time</p>
						<p class="text-3xl font-bold text-foreground">{{ formatMs(summary.avgProcessingTime) }}</p>
						<p class="text-xs text-success mt-2">Based on {{ summary.completed }} runs</p>
					</div>
					<div class="pb-4 border-b border-border/20">
						<p class="text-foreground/70 text-sm font-medium mb-2">Total Data Processed</p>
						<p class="text-3xl font-bold text-foreground">{{ formatBytes(summary.totalDataProcessed) }}</p>
						<p class="text-xs text-success mt-2">{{ summary.total }} files ingested</p>
					</div>
					<div class="pb-4 border-b border-border/20">
						<p class="text-foreground/70 text-sm font-medium mb-2">Success Rate</p>
						<p class="text-3xl font-bold text-foreground">{{ summary.successRate.toFixed(1) }}%</p>
						<p class="text-xs text-foreground/60 mt-2">{{ summary.failed }} failed runs</p>
					</div>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped></style>
