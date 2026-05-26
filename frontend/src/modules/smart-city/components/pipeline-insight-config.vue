<script setup lang="ts">
import { computed, ref } from 'vue'
import { BrainCircuit, ChevronDown, Clock3, Play, Settings2, Sparkles } from 'lucide-vue-next'
import type { PipelineInsightConfig } from '@/api/client'

const props = defineProps<{
	config: PipelineInsightConfig | null
	loading: boolean
	running: boolean
}>()

const emit = defineEmits<{
	save: [input: Partial<Pick<PipelineInsightConfig, 'intervalMinutes' | 'depth' | 'focus'>>]
	runNow: []
}>()

const settingsOpen = ref(false)

const intervalMinutes = computed({
	get: () => props.config?.intervalMinutes ?? 0,
	set: value => emit('save', { intervalMinutes: Number(value) }),
})

const depth = computed({
	get: () => props.config?.depth ?? 'standard',
	set: value => emit('save', { depth: value as PipelineInsightConfig['depth'] }),
})

const focus = computed({
	get: () => props.config?.focus ?? 'all',
	set: value => emit('save', { focus: value as PipelineInsightConfig['focus'] }),
})

const scheduleLabel = computed(() => {
	const minutes = props.config?.intervalMinutes ?? 0
	if (minutes === 0) return 'Manual only'
	if (minutes === 15) return 'Every 15 min'
	if (minutes === 30) return 'Every 30 min'
	if (minutes === 60) return 'Hourly'
	if (minutes === 360) return 'Every 6 h'
	if (minutes === 1440) return 'Daily'
	return `Every ${minutes} min`
})

const depthLabel = computed(() => {
	const labels: Record<string, string> = {
		quick: 'Quick scan',
		standard: 'Standard',
		deep: 'Deep research',
	}
	return labels[props.config?.depth ?? 'standard'] ?? 'Standard'
})

const focusLabel = computed(() => {
	const labels: Record<string, string> = {
		all: 'All signals',
		ops: 'Operations',
		quality: 'Data quality',
		geo: 'Geospatial',
		exports: 'Exports',
	}
	return labels[props.config?.focus ?? 'all'] ?? 'All signals'
})
</script>

<template>
	<section class="pipeline-insight-config" :class="{ 'pipeline-insight-config--expanded': settingsOpen }">
		<div class="pipeline-insight-config__bar">
			<div class="pipeline-insight-config__bar-left">
				<div class="pipeline-insight-config__icon">
					<BrainCircuit class="w-4 h-4" />
				</div>
				<div class="pipeline-insight-config__summary">
					<span class="pipeline-insight-config__summary-label">Analysis engine</span>
					<div class="pipeline-insight-config__pills">
						<span class="pipeline-insight-config__pill">{{ scheduleLabel }}</span>
						<span class="pipeline-insight-config__pill">{{ depthLabel }}</span>
						<span class="pipeline-insight-config__pill">{{ focusLabel }}</span>
					</div>
				</div>
			</div>

			<div class="pipeline-insight-config__bar-actions">
				<div class="pipeline-insight-config__meta">
					<Clock3 class="w-3.5 h-3.5" />
					<span v-if="config?.lastRunAt">Last {{ new Date(config.lastRunAt).toLocaleString() }}</span>
					<span v-else>No runs yet</span>
				</div>
				<button
					type="button"
					class="pipeline-insight-config__settings-toggle"
					:aria-expanded="settingsOpen"
					@click="settingsOpen = !settingsOpen"
				>
					<Settings2 class="w-3.5 h-3.5" />
					Settings
					<ChevronDown class="pipeline-insight-config__settings-chevron" />
				</button>
				<button type="button" class="pipeline-insight-config__run" :disabled="loading || running" @click="emit('runNow')">
					<Play v-if="!running" class="w-4 h-4" />
					<Sparkles v-else class="w-4 h-4 animate-pulse" />
					{{ running ? 'Analyzing…' : 'Run now' }}
				</button>
			</div>
		</div>

		<div v-show="settingsOpen" class="pipeline-insight-config__drawer">
			<p class="pipeline-insight-config__desc">
				Profiles your data, detects patterns, picks adaptive visualizations, and narrates findings on a schedule.
			</p>
			<div class="pipeline-insight-config__grid">
				<label class="pipeline-insight-config__field">
					<span>Schedule</span>
					<select v-model="intervalMinutes" :disabled="loading">
						<option :value="0">Off</option>
						<option :value="15">Every 15 minutes</option>
						<option :value="30">Every 30 minutes</option>
						<option :value="60">Every hour</option>
						<option :value="360">Every 6 hours</option>
						<option :value="1440">Daily</option>
					</select>
				</label>
				<label class="pipeline-insight-config__field">
					<span>Depth</span>
					<select v-model="depth" :disabled="loading">
						<option value="quick">Quick scan</option>
						<option value="standard">Standard</option>
						<option value="deep">Deep research</option>
					</select>
				</label>
				<label class="pipeline-insight-config__field">
					<span>Focus</span>
					<select v-model="focus" :disabled="loading">
						<option value="all">All signals</option>
						<option value="ops">Operations</option>
						<option value="quality">Data quality</option>
						<option value="geo">Geospatial</option>
						<option value="exports">Exports</option>
					</select>
				</label>
			</div>
			<p v-if="config?.nextRunAt && config.enabled" class="pipeline-insight-config__next-run">
				Next scheduled run {{ new Date(config.nextRunAt).toLocaleString() }}
			</p>
		</div>
	</section>
</template>
