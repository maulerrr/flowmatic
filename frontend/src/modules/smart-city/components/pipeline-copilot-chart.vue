<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import { Maximize2, X } from 'lucide-vue-next'
import type { CopilotVizSpec } from '@/api/client'
import PipelineCopilotChartBody from './pipeline-copilot-chart-body.vue'
import PipelineCopilotGeoMap from './pipeline-copilot-geo-map.vue'

const props = defineProps<{ spec: CopilotVizSpec }>()

const expanded = ref(false)
const modalGeoMap = ref<InstanceType<typeof PipelineCopilotGeoMap> | null>(null)

async function openModal() {
	expanded.value = true
	await nextTick()
	await nextTick()
	await modalGeoMap.value?.refreshMap?.()
}

function closeModal() {
	expanded.value = false
}

function onKeydown(event: KeyboardEvent) {
	if (event.key === 'Escape' && expanded.value) closeModal()
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onUnmounted(() => window.removeEventListener('keydown', onKeydown))
</script>

<template>
	<article class="copilot-chart copilot-chart--clickable" role="button" tabindex="0" @click="openModal()" @keydown.enter="openModal()">
		<header class="copilot-chart__header">
			<div>
				<h4 class="copilot-chart__title">{{ spec.title }}</h4>
				<p v-if="spec.subtitle" class="copilot-chart__subtitle">{{ spec.subtitle }}</p>
			</div>
			<div class="copilot-chart__header-actions">
				<span class="copilot-chart__type">{{ spec.type }}</span>
				<span class="copilot-chart__expand-hint"><Maximize2 class="w-3.5 h-3.5" /> Expand</span>
			</div>
		</header>

		<PipelineCopilotGeoMap v-if="spec.type === 'map'" :spec="spec" height="12rem" :interactive="false" />
		<PipelineCopilotChartBody v-else :spec="spec" compact />
	</article>

	<Teleport to="body">
		<div v-if="expanded" class="copilot-chart-modal" @click.self="closeModal()">
			<div class="copilot-chart-modal__panel">
				<header class="copilot-chart-modal__header">
					<div>
						<h3 class="copilot-chart-modal__title">{{ spec.title }}</h3>
						<p v-if="spec.subtitle" class="copilot-chart-modal__subtitle">{{ spec.subtitle }}</p>
					</div>
					<button type="button" class="copilot-chart-modal__close" @click="closeModal()">
						<X class="w-5 h-5" />
					</button>
				</header>
				<div class="copilot-chart-modal__body">
					<PipelineCopilotGeoMap
						v-if="spec.type === 'map'"
						ref="modalGeoMap"
						:spec="spec"
						height="min(68vh, 620px)"
					/>
					<PipelineCopilotChartBody v-else :spec="spec" />
				</div>
			</div>
		</div>
	</Teleport>
</template>
