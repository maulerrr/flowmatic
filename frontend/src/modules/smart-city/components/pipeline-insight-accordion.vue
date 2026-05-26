<script setup lang="ts">
import { ChevronDown } from 'lucide-vue-next'

defineProps<{
	title: string
	subtitle?: string
	count?: number
	open: boolean
	tone?: 'default' | 'warning' | 'critical'
}>()

const emit = defineEmits<{ toggle: [] }>()
</script>

<template>
	<div
		class="insight-accordion"
		:class="[
			`insight-accordion--${tone ?? 'default'}`,
			{ 'insight-accordion--open': open },
		]"
	>
		<button type="button" class="insight-accordion__trigger" @click="emit('toggle')">
			<span class="insight-accordion__trigger-main">
				<span class="insight-accordion__title">{{ title }}</span>
				<span v-if="count !== undefined" class="insight-accordion__count">{{ count }}</span>
			</span>
			<span v-if="subtitle" class="insight-accordion__subtitle">{{ subtitle }}</span>
			<ChevronDown class="insight-accordion__chevron" />
		</button>
		<div v-show="open" class="insight-accordion__panel">
			<slot />
		</div>
	</div>
</template>
