<script setup lang="ts">
import { inject } from 'vue'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<div
		class="pipeline-live-conduit"
		:class="{ 'pipeline-live-conduit--active': pipeline.isPipelineLive }"
		aria-hidden="true"
	>
		<svg class="pipeline-live-conduit__svg" viewBox="0 0 1200 120" preserveAspectRatio="none">
			<defs>
				<linearGradient id="conduit-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
					<stop offset="0%" stop-color="rgba(52, 208, 195, 0)" />
					<stop offset="45%" stop-color="rgba(52, 208, 195, 0.85)" />
					<stop offset="55%" stop-color="rgba(124, 140, 255, 0.85)" />
					<stop offset="100%" stop-color="rgba(124, 140, 255, 0)" />
				</linearGradient>
			</defs>
			<path
				class="pipeline-live-conduit__track"
				d="M 60 60 H 340 M 360 60 H 640 M 660 60 H 940 M 960 60 H 1140"
			/>
			<path
				class="pipeline-live-conduit__pulse pipeline-live-conduit__pulse--a"
				d="M 60 60 H 340 M 360 60 H 640 M 660 60 H 940 M 960 60 H 1140"
			/>
			<path
				class="pipeline-live-conduit__pulse pipeline-live-conduit__pulse--b"
				d="M 60 60 H 340 M 360 60 H 640 M 660 60 H 940 M 960 60 H 1140"
			/>
		</svg>
		<div class="pipeline-live-conduit__nodes">
			<span v-for="index in 4" :key="index" class="pipeline-live-conduit__node-ring" />
		</div>
	</div>
</template>
