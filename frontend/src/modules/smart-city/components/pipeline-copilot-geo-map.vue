<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { CopilotVizSpec } from '@/api/client'

const props = withDefaults(
	defineProps<{
		spec: CopilotVizSpec
		height?: string
		interactive?: boolean
	}>(),
	{
		height: '12rem',
		interactive: true,
	},
)

const containerRef = ref<HTMLElement | null>(null)
let map: L.Map | null = null
let layerGroup: L.LayerGroup | null = null

const isHeatmap = computed(
	() => props.spec.type === 'map' && props.spec.meta?.mode === 'heatmap',
)

const bbox = computed(() => {
	const raw = props.spec.meta?.bbox as
		| { minLat: number; maxLat: number; minLng: number; maxLng: number }
		| undefined
	return (
		raw ?? {
			minLat: 51.08,
			maxLat: 51.2,
			minLng: 71.34,
			maxLng: 71.57,
		}
	)
})

const points = computed(() => {
	if (isHeatmap.value) return []
	return ((props.spec.meta?.points as Array<{ lat: number; lng: number; weight?: number }>) ?? []).filter(
		point => Number.isFinite(point.lat) && Number.isFinite(point.lng),
	)
})

const hotspots = computed(() => {
	if (!isHeatmap.value) return []
	return (
		(props.spec.meta?.hotspots as Array<{
			lat: number
			lng: number
			intensity?: number
			weight?: number
			label?: string
		}>) ?? []
	).filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lng))
})

function renderLayers() {
	if (!map || !layerGroup) return
	layerGroup.clearLayers()

	if (isHeatmap.value) {
		const maxIntensity = Math.max(...hotspots.value.map(item => item.intensity ?? item.weight ?? 1), 1)
		for (const hotspot of hotspots.value) {
			const intensity = hotspot.intensity ?? hotspot.weight ?? 1
			L.circle([hotspot.lat, hotspot.lng], {
				radius: 180 + (intensity / maxIntensity) * 520,
				color: '#f2c14f',
				fillColor: '#f2c14f',
				fillOpacity: 0.18 + (intensity / maxIntensity) * 0.42,
				weight: 1,
			})
				.bindTooltip(hotspot.label ?? 'Activity cluster')
				.addTo(layerGroup)
		}
	} else {
		for (const point of points.value) {
			L.circleMarker([point.lat, point.lng], {
				radius: 5,
				color: '#34d0c3',
				fillColor: '#34d0c3',
				fillOpacity: 0.85,
				weight: 1,
			}).addTo(layerGroup)
		}
	}

	const bounds = L.latLngBounds([
		[bbox.value.minLat, bbox.value.minLng],
		[bbox.value.maxLat, bbox.value.maxLng],
	])
	map.fitBounds(bounds.pad(0.08), { animate: false })
}

function initMap() {
	if (!containerRef.value || map) return
	map = L.map(containerRef.value, {
		zoomControl: props.interactive,
		scrollWheelZoom: props.interactive,
		dragging: props.interactive,
		doubleClickZoom: props.interactive,
		attributionControl: true,
	})
	L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
		maxZoom: 18,
		attribution: '&copy; OpenStreetMap contributors',
	}).addTo(map)
	layerGroup = L.layerGroup().addTo(map)
	renderLayers()
}

function destroyMap() {
	if (map) {
		map.remove()
		map = null
		layerGroup = null
	}
}

async function refreshMap() {
	if (!map) {
		initMap()
		return
	}
	await nextTick()
	renderLayers()
	map.invalidateSize()
}

onMounted(() => {
	void nextTick(() => initMap())
})

onUnmounted(() => {
	destroyMap()
})

watch(
	() => [props.spec, props.height, props.interactive],
	() => {
		void refreshMap()
	},
	{ deep: true },
)

defineExpose({ refreshMap })
</script>

<template>
	<div class="copilot-geo-map" :style="{ height }">
		<div ref="containerRef" class="copilot-geo-map__canvas" />
	</div>
</template>
