<script setup lang="ts">
import { computed } from 'vue'
import type { CopilotVizSpec } from '@/api/client'

const props = withDefaults(defineProps<{ spec: CopilotVizSpec; compact?: boolean }>(), {
	compact: false,
})

const maxValue = computed(() => {
	const values = props.spec.series.flatMap(series => series.values)
	return Math.max(...values, 1)
})

const donutStyle = computed(() => {
	const values = props.spec.series[0]?.values ?? []
	const total = values.reduce((sum, value) => sum + value, 0) || 1
	const colors = ['#34d0c3', '#f87171', '#7c8cff']
	let cursor = 0
	const segments = values.map((value, index) => {
		const start = cursor
		const slice = (value / total) * 100
		cursor += slice
		return `${colors[index % colors.length]} ${start}% ${cursor}%`
	})
	return { background: `conic-gradient(${segments.join(', ')})` }
})

function barHeight(value: number) {
	return `${Math.max(8, (value / maxValue.value) * 100)}%`
}

function normalizeSeries(values: number[]) {
	if (values.length === 0) return []
	if (values.length === 1) return [values[0], values[0]]
	return values
}

function getAreaPath(values: number[]) {
	const normalized = normalizeSeries(values)
	if (normalized.length === 0) return { area: '', line: '' }
	const width = 100
	const height = 100
	const step = normalized.length > 1 ? width / (normalized.length - 1) : width
	const points = normalized.map((value, index) => {
		const x = index * step
		const y = height - (value / maxValue.value) * (height - 8) - 4
		return [x, y] as const
	})
	const line = points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'} ${x} ${y}`).join(' ')
	const area = `${line} L ${width} ${height} L 0 ${height} Z`
	return { area, line }
}

const areaPaths = computed(() => getAreaPath(props.spec.series[0]?.values ?? []))
const tableColumns = computed(() => {
	const row = props.spec.rows?.[0]
	return row ? Object.keys(row) : []
})

const axisLabels = computed(() => {
	const labels = props.spec.labels
	if (labels.length <= 6) return labels
	const step = Math.max(1, Math.floor(labels.length / 5))
	return labels.filter((_, index) => index % step === 0 || index === labels.length - 1)
})

const scatterPoints = computed(() => {
	const points = (props.spec.meta?.points as Array<{ x: number; y: number }>) ?? []
	if (!points.length) return []
	const xs = points.map(point => point.x)
	const ys = points.map(point => point.y)
	const minX = Math.min(...xs)
	const maxX = Math.max(...xs)
	const minY = Math.min(...ys)
	const maxY = Math.max(...ys)
	const spanX = Math.max(maxX - minX, 1)
	const spanY = Math.max(maxY - minY, 1)
	return points.map(point => ({
		x: ((point.x - minX) / spanX) * 100,
		y: 100 - ((point.y - minY) / spanY) * 100,
	}))
})

const timelinePaths = computed(() => props.spec.series.map(series => getAreaPath(series.values)))

const plotClass = computed(() => ({
	'copilot-chart__plot--compact': props.compact,
	'copilot-chart__plot--expanded': !props.compact,
}))
</script>

<template>
	<div v-if="spec.type === 'kpi'" class="copilot-chart__kpis">
		<div v-for="kpi in spec.kpis ?? []" :key="kpi.label" class="copilot-chart__kpi">
			<div class="copilot-chart__kpi-label">{{ kpi.label }}</div>
			<div class="copilot-chart__kpi-value">{{ kpi.value }}</div>
		</div>
	</div>

	<div v-else-if="spec.type === 'table'" class="copilot-chart__table-wrap">
		<table class="copilot-chart__table">
			<thead>
				<tr>
					<th v-for="column in tableColumns" :key="column">{{ column }}</th>
				</tr>
			</thead>
			<tbody>
				<tr v-for="(row, index) in spec.rows ?? []" :key="index">
					<td v-for="column in tableColumns" :key="column">{{ row[column] ?? '—' }}</td>
				</tr>
			</tbody>
		</table>
	</div>

	<div v-else-if="spec.type === 'donut'" class="copilot-chart__donut-wrap">
		<div class="copilot-chart__donut" :style="donutStyle">
			<div class="copilot-chart__donut-hole">
				<span class="copilot-chart__donut-total">
					{{ (spec.series[0]?.values ?? []).reduce((sum, value) => sum + value, 0) }}
				</span>
				<span class="copilot-chart__donut-label">total</span>
			</div>
		</div>
		<ul class="copilot-chart__legend">
			<li v-for="(label, index) in spec.labels" :key="label">
				<span>{{ label }}</span>
				<strong>{{ spec.series[0]?.values[index] ?? 0 }}</strong>
			</li>
		</ul>
	</div>

	<div v-else-if="spec.type === 'funnel'" class="copilot-chart__funnel">
		<div
			v-for="(label, index) in spec.labels"
			:key="label"
			class="copilot-chart__funnel-step"
			:style="{ width: `${Math.max(28, 100 - index * 18)}%` }"
		>
			<span>{{ label }}</span>
			<strong>{{ spec.series[0]?.values[index]?.toLocaleString() ?? 0 }}</strong>
		</div>
	</div>

	<div v-else-if="spec.type === 'heatmap'" class="copilot-chart__plot copilot-chart__plot--bar" :class="plotClass">
		<div class="copilot-chart__bars">
			<div v-for="(label, index) in spec.labels" :key="label" class="copilot-chart__bar-col">
				<div
					class="copilot-chart__bar"
					:style="{
						height: barHeight(spec.series[0]?.values[index] ?? 0),
						background: `linear-gradient(180deg, #f2c14f, rgba(242,193,79,0.25))`,
					}"
				/>
				<span>{{ label }}</span>
			</div>
		</div>
	</div>

	<div v-else-if="spec.type === 'scatter'" class="copilot-chart__plot copilot-chart__plot--scatter" :class="plotClass">
		<svg viewBox="0 0 100 100" class="copilot-chart__svg">
			<rect x="0" y="0" width="100" height="100" rx="8" fill="rgba(15,26,40,0.85)" />
			<circle
				v-for="(point, index) in scatterPoints"
				:key="index"
				:cx="point.x"
				:cy="point.y"
				r="1.4"
				fill="#7c8cff"
				opacity="0.8"
			/>
		</svg>
		<div class="copilot-chart__axis">
			<span>{{ spec.meta?.xField ?? 'x' }}</span>
			<span>{{ spec.meta?.yField ?? 'y' }}</span>
		</div>
	</div>

	<div v-else-if="spec.type === 'timeline'" class="copilot-chart__plot copilot-chart__plot--timeline" :class="plotClass">
		<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="copilot-chart__svg">
			<path
				v-for="(series, index) in spec.series"
				:key="series.name"
				:d="timelinePaths[index]?.line"
				fill="none"
				:stroke="series.color ?? '#7c8cff'"
				stroke-width="1.5"
				vector-effect="non-scaling-stroke"
			/>
		</svg>
		<ul class="copilot-chart__legend copilot-chart__legend--timeline">
			<li v-for="series in spec.series" :key="series.name">
				<span class="copilot-chart__legend-dot" :style="{ background: series.color ?? '#7c8cff' }" />
				<span>{{ series.name }}</span>
				<strong>{{ series.values.reduce((sum, value) => sum + value, 0) }}</strong>
			</li>
		</ul>
		<div class="copilot-chart__axis copilot-chart__axis--timeline">
			<span v-for="label in axisLabels" :key="label">{{ label }}</span>
		</div>
	</div>

	<div v-else-if="spec.type === 'area'" class="copilot-chart__plot" :class="plotClass">
		<svg viewBox="0 0 100 100" preserveAspectRatio="none" class="copilot-chart__svg">
			<defs>
				<linearGradient id="copilot-area-fill" x1="0" x2="0" y1="0" y2="1">
					<stop offset="0%" stop-color="#34d0c3" stop-opacity="0.45" />
					<stop offset="100%" stop-color="#34d0c3" stop-opacity="0.03" />
				</linearGradient>
			</defs>
			<path :d="areaPaths.area" fill="url(#copilot-area-fill)" />
			<path :d="areaPaths.line" fill="none" stroke="#34d0c3" stroke-width="1.5" vector-effect="non-scaling-stroke" />
		</svg>
		<div class="copilot-chart__axis">
			<span v-for="label in axisLabels" :key="label">{{ label }}</span>
		</div>
	</div>

	<div v-else class="copilot-chart__plot copilot-chart__plot--bar" :class="plotClass">
		<div class="copilot-chart__bars">
			<div v-for="(label, index) in spec.labels" :key="label" class="copilot-chart__bar-col">
				<div
					class="copilot-chart__bar"
					:style="{
						height: barHeight(spec.series[0]?.values[index] ?? 0),
						background: spec.series[0]?.color ?? '#7c8cff',
					}"
				/>
				<span>{{ label }}</span>
			</div>
		</div>
	</div>

	<div v-if="spec.kpis?.length && spec.type !== 'kpi'" class="copilot-chart__footer-kpis">
		<div v-for="kpi in spec.kpis" :key="kpi.label">
			<span>{{ kpi.label }}</span>
			<strong>{{ kpi.value }}</strong>
		</div>
	</div>
</template>
