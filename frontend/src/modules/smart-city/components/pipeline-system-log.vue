<script setup lang="ts">
import { inject } from 'vue'
import { PipelineWorkbenchKey } from '../pipeline-context'

const pipeline = inject(PipelineWorkbenchKey)
if (!pipeline) throw new Error('PipelineWorkbenchKey is not provided.')
</script>

<template>
	<section class="pipeline-log">
		<div class="flex justify-between items-center mb-2 pb-2 border-b border-white/5">
			<span class="text-gray-500 uppercase font-bold tracking-wider">System Log</span>
			<span class="text-gray-600">{{ pipeline.loading ? 'syncing' : 'ready' }}</span>
		</div>
		<div class="flex-1 overflow-y-auto space-y-1">
			<div v-for="(log, i) in pipeline.logs" :key="i"><span class="text-primary mr-2">&gt;</span>{{ log }}</div>
			<div v-if="pipeline.logs.length === 0" class="text-gray-600">No runtime messages yet.</div>
		</div>
	</section>
</template>
