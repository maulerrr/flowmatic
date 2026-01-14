<script setup lang="ts" generic="T extends Record<string, any>">
import { LucideIcon } from 'lucide-vue-next';





interface Props {
	label: string
	value: string | number
	icon: typeof LucideIcon
	color: 'primary' | 'success' | 'warning' | 'destructive'
	description?: string
	trend?: {
		value: number
		positive: boolean
	}
}

withDefaults(defineProps<Props>(), {})

const colorMap = {
	primary: { bg: 'bg-primary/10', text: 'text-primary' },
	success: { bg: 'bg-success/10', text: 'text-success' },
	warning: { bg: 'bg-warning/10', text: 'text-warning' },
	destructive: { bg: 'bg-destructive/10', text: 'text-destructive' },
}
</script>

<template>
	<div
		class="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-5 hover:border-primary/40 transition-all group"
	>
		<div class="flex items-start justify-between mb-3">
			<div
				:class="[colorMap[color].bg, 'w-10 h-10 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform']"
			>
				<component
					:is="icon"
					:class="['w-5 h-5', colorMap[color].text]"
				/>
			</div>
		</div>
		<p class="text-foreground/70 text-xs font-medium uppercase tracking-wider mb-1">{{ label }}</p>
		<div class="flex items-baseline gap-2">
			<p class="text-2xl font-bold text-foreground">{{ value }}</p>
			<p
				v-if="description"
				class="text-xs text-foreground/60"
			>
				{{ description }}
			</p>
		</div>
		<div
			v-if="trend"
			class="mt-2 pt-2 border-t border-border/20 flex items-center gap-1.5"
		>
			<span
				:class="['text-xs font-semibold', trend.positive ? 'text-success' : 'text-destructive']"
			>
				{{ trend.positive ? '↑' : '↓' }} {{ Math.abs(trend.value) }}%
			</span>
		</div>
	</div>
</template>

<style scoped></style>
