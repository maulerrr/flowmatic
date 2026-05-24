<script setup lang="ts">
import { inject } from 'vue'
import { Radio } from 'lucide-vue-next'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section class="pipeline-empty pipeline-animate-rise">
		<Radio class="w-10 h-10 mx-auto text-primary" />
		<h2 class="text-lg font-semibold text-foreground">Create your first real-time pipeline</h2>
		<p class="text-sm text-foreground/60">Sources, events, data lake settings, and workflow JSON persist per organization.</p>
		<button @click="pipeline.showPipelineModal = true" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium">
			Create Pipeline
		</button>
	</section>
</template>
