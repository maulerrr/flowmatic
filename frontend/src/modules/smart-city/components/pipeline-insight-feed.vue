<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import {
	AlertTriangle,
	BarChart3,
	BookOpen,
	CheckCircle2,
	ChevronRight,
	Info,
	Lightbulb,
	MapPin,
	TrendingUp,
	X,
} from 'lucide-vue-next'
import type { PipelineInsightRun } from '@/api/client'
import PipelineCopilotChart from './pipeline-copilot-chart.vue'
import PipelineInsightAccordion from './pipeline-insight-accordion.vue'

const props = defineProps<{
	runs: PipelineInsightRun[]
	selectedRunId: string
	loading: boolean
}>()

const emit = defineEmits<{
	selectRun: [runId: string]
}>()

const selectedRun = computed(() => props.runs.find(run => run.id === props.selectedRunId) ?? props.runs[0] ?? null)

const briefModalOpen = ref(false)
const openSections = ref({ brief: false, findings: false, actions: false })

const criticalCount = computed(
	() => selectedRun.value?.findings.filter(finding => finding.severity === 'critical').length ?? 0,
)
const warningCount = computed(
	() => selectedRun.value?.findings.filter(finding => finding.severity === 'warning').length ?? 0,
)

const findingsTone = computed(() => {
	if (criticalCount.value > 0) return 'critical'
	if (warningCount.value > 0) return 'warning'
	return 'default'
})

const findingsSubtitle = computed(() => {
	if (!selectedRun.value?.findings.length) return ''
	const parts: string[] = []
	if (criticalCount.value) parts.push(`${criticalCount.value} critical`)
	if (warningCount.value) parts.push(`${warningCount.value} warning`)
	if (parts.length) return parts.join(' · ')
	return selectedRun.value.findings[0]?.title ?? ''
})

const narrativePreview = computed(() => {
	const text = selectedRun.value?.narrative?.trim() ?? ''
	if (!text) return ''
	const firstParagraph = text.split('\n').find(line => line.trim()) ?? text
	if (firstParagraph.length <= 180) return firstParagraph
	return `${firstParagraph.slice(0, 180).trim()}…`
})

const hasLongNarrative = computed(() => (selectedRun.value?.narrative?.length ?? 0) > 180)

watch(
	selectedRun,
	run => {
		openSections.value = {
			brief: false,
			findings: (run?.findings.some(finding => finding.severity === 'critical') ?? false),
			actions: false,
		}
		briefModalOpen.value = false
	},
	{ immediate: true },
)

function severityIcon(severity: string) {
	if (severity === 'critical' || severity === 'warning') return AlertTriangle
	return Info
}

function categoryIcon(category: string) {
	if (category === 'geo') return MapPin
	if (category === 'correlation' || category === 'pattern') return TrendingUp
	return Lightbulb
}

function toggleSection(section: keyof typeof openSections.value) {
	openSections.value[section] = !openSections.value[section]
}

function scrollToCharts() {
	document.getElementById('insight-charts')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function onBriefKeydown(event: KeyboardEvent) {
	if (event.key === 'Escape' && briefModalOpen.value) briefModalOpen.value = false
}

onMounted(() => window.addEventListener('keydown', onBriefKeydown))
onUnmounted(() => window.removeEventListener('keydown', onBriefKeydown))
</script>

<template>
	<section class="pipeline-insight-feed">
		<div class="pipeline-insight-feed__header">
			<div>
				<h3 class="pipeline-insight-feed__title">Latest analysis</h3>
				<p class="pipeline-insight-feed__hint">Charts up front — details on demand.</p>
			</div>
			<select
				v-if="runs.length"
				:value="selectedRun?.id"
				class="pipeline-insight-feed__select"
				@change="emit('selectRun', ($event.target as HTMLSelectElement).value)"
			>
				<option v-for="run in runs" :key="run.id" :value="run.id">
					{{ new Date(run.createdAt).toLocaleString() }} · {{ run.trigger }} · {{ run.status }}
				</option>
			</select>
		</div>

		<div v-if="loading && !runs.length" class="pipeline-insight-feed__empty">Loading insight history…</div>
		<div v-else-if="!selectedRun" class="pipeline-insight-feed__empty">
			No insight runs yet. Enable a schedule or click “Run now”.
		</div>

		<article v-else class="pipeline-insight-run">
			<header class="pipeline-insight-run__hero">
				<div class="pipeline-insight-run__hero-main">
					<div class="pipeline-insight-run__eyebrow">
						<CheckCircle2 v-if="selectedRun.status === 'SUCCEEDED'" class="w-3.5 h-3.5" />
						<AlertTriangle v-else class="w-3.5 h-3.5" />
						{{ selectedRun.plannerSource === 'llm' ? 'AI planner' : 'Rule planner' }}
						· {{ selectedRun.depth }}
						· {{ selectedRun.focus }}
					</div>
					<h4 class="pipeline-insight-run__headline">{{ selectedRun.headline }}</h4>
					<p v-if="narrativePreview && !selectedRun.errorMessage" class="pipeline-insight-run__teaser">
						{{ narrativePreview }}
					</p>
					<p v-else-if="selectedRun.errorMessage" class="pipeline-insight-run__error">{{ selectedRun.errorMessage }}</p>
				</div>
				<span
					class="pipeline-insight-run__status"
					:class="`pipeline-insight-run__status--${selectedRun.status.toLowerCase()}`"
				>
					{{ selectedRun.status }}
				</span>
			</header>

			<div v-if="selectedRun.profile.domains?.length" class="pipeline-insight-run__domains">
				<span v-for="domain in selectedRun.profile.domains" :key="domain" class="pipeline-insight-run__domain">
					{{ domain }}
				</span>
			</div>

			<div class="pipeline-insight-run__stat-rail">
				<button
					v-if="selectedRun.visualizations.length"
					type="button"
					class="pipeline-insight-run__stat"
					@click="scrollToCharts()"
				>
					<BarChart3 class="w-4 h-4" />
					<span class="pipeline-insight-run__stat-value">{{ selectedRun.visualizations.length }}</span>
					<span class="pipeline-insight-run__stat-label">Charts</span>
				</button>
				<button
					v-if="selectedRun.findings.length"
					type="button"
					class="pipeline-insight-run__stat"
					:class="{ 'pipeline-insight-run__stat--active': openSections.findings }"
					@click="toggleSection('findings')"
				>
					<AlertTriangle v-if="findingsTone !== 'default'" class="w-4 h-4" />
					<Lightbulb v-else class="w-4 h-4" />
					<span class="pipeline-insight-run__stat-value">{{ selectedRun.findings.length }}</span>
					<span class="pipeline-insight-run__stat-label">Findings</span>
				</button>
				<button
					v-if="selectedRun.actions.length"
					type="button"
					class="pipeline-insight-run__stat"
					:class="{ 'pipeline-insight-run__stat--active': openSections.actions }"
					@click="toggleSection('actions')"
				>
					<ChevronRight class="w-4 h-4" />
					<span class="pipeline-insight-run__stat-value">{{ selectedRun.actions.length }}</span>
					<span class="pipeline-insight-run__stat-label">Actions</span>
				</button>
				<button
					v-if="selectedRun.narrative"
					type="button"
					class="pipeline-insight-run__stat pipeline-insight-run__stat--brief"
					@click="briefModalOpen = true"
				>
					<BookOpen class="w-4 h-4" />
					<span class="pipeline-insight-run__stat-label">Full brief</span>
				</button>
			</div>

			<div v-if="selectedRun.visualizations.length" id="insight-charts" class="pipeline-insight-run__viz-section">
				<div class="pipeline-insight-run__viz-header">
					<h5>Visual intelligence</h5>
					<span>Click any chart to expand</span>
				</div>
				<div class="pipeline-insight-run__viz-grid">
					<PipelineCopilotChart
						v-for="visualization in selectedRun.visualizations"
						:key="visualization.id"
						:spec="visualization"
					/>
				</div>
			</div>

			<div class="pipeline-insight-run__details">
				<PipelineInsightAccordion
					v-if="selectedRun.narrative"
					title="Analysis brief"
					:subtitle="hasLongNarrative ? 'Tap to read on page' : undefined"
					:open="openSections.brief"
					@toggle="toggleSection('brief')"
				>
					<p class="pipeline-insight-run__narrative">{{ selectedRun.narrative }}</p>
					<button
						v-if="hasLongNarrative"
						type="button"
						class="pipeline-insight-run__read-modal"
						@click="briefModalOpen = true"
					>
						Open in reading view
					</button>
				</PipelineInsightAccordion>

				<PipelineInsightAccordion
					v-if="selectedRun.findings.length"
					title="Findings"
					:subtitle="findingsSubtitle"
					:count="selectedRun.findings.length"
					:open="openSections.findings"
					:tone="findingsTone"
					@toggle="toggleSection('findings')"
				>
					<div class="pipeline-insight-run__findings">
						<div
							v-for="finding in selectedRun.findings"
							:key="finding.id"
							class="pipeline-insight-run__finding"
							:class="`pipeline-insight-run__finding--${finding.severity}`"
						>
							<component :is="categoryIcon(finding.category)" class="pipeline-insight-run__finding-icon" />
							<div>
								<div class="pipeline-insight-run__finding-title">
									<component :is="severityIcon(finding.severity)" class="w-3 h-3 opacity-70" />
									{{ finding.title }}
								</div>
								<p>{{ finding.summary }}</p>
							</div>
						</div>
					</div>
				</PipelineInsightAccordion>

				<PipelineInsightAccordion
					v-if="selectedRun.actions.length"
					title="Recommended actions"
					:subtitle="selectedRun.actions[0]?.label"
					:count="selectedRun.actions.length"
					:open="openSections.actions"
					@toggle="toggleSection('actions')"
				>
					<div class="pipeline-insight-run__actions">
						<div v-for="action in selectedRun.actions" :key="action.id" class="pipeline-insight-run__action">
							<strong>{{ action.label }}</strong>
							<p>{{ action.description }}</p>
						</div>
					</div>
				</PipelineInsightAccordion>
			</div>
		</article>
	</section>

	<Teleport to="body">
		<div v-if="briefModalOpen && selectedRun?.narrative" class="insight-brief-modal" @click.self="briefModalOpen = false">
			<div class="insight-brief-modal__panel">
				<header class="insight-brief-modal__header">
					<div>
						<div class="insight-brief-modal__eyebrow">Analysis brief</div>
						<h3 class="insight-brief-modal__title">{{ selectedRun.headline }}</h3>
						<p class="insight-brief-modal__meta">
							{{ new Date(selectedRun.createdAt).toLocaleString() }}
							· {{ selectedRun.plannerSource === 'llm' ? 'AI planner' : 'Rule planner' }}
						</p>
					</div>
					<button type="button" class="insight-brief-modal__close" @click="briefModalOpen = false">
						<X class="w-5 h-5" />
					</button>
				</header>
				<div class="insight-brief-modal__body">
					<p class="pipeline-insight-run__narrative">{{ selectedRun.narrative }}</p>
					<div v-if="selectedRun.findings.length" class="insight-brief-modal__findings">
						<h4>Key findings</h4>
						<ul>
							<li v-for="finding in selectedRun.findings.slice(0, 5)" :key="finding.id">
								<strong>{{ finding.title }}</strong>
								<span>{{ finding.summary }}</span>
							</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	</Teleport>
</template>
