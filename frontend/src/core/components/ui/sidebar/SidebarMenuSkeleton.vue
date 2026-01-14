<script setup lang="ts">
import { type HTMLAttributes, computed } from 'vue';



import { Skeleton } from '@/core/components/ui/skeleton';
import { cn } from '@/core/utils/tailwind.utils';





const props = defineProps<{
  showIcon?: boolean
  class?: HTMLAttributes['class']
}>()

const width = computed(() => {
  return `${Math.floor(Math.random() * 40) + 50}%`;
})
</script>

<template>
	<div
		data-slot="sidebar-menu-skeleton"
		data-sidebar="menu-skeleton"
		:class="cn('flex h-8 items-center gap-2 rounded-md px-2', props.class)"
	>
		<Skeleton
			v-if="showIcon"
			class="rounded-md size-4"
			data-sidebar="menu-skeleton-icon"
		/>

		<Skeleton
			class="flex-1 max-w-(--skeleton-width) h-4"
			data-sidebar="menu-skeleton-text"
			:style="{ '--skeleton-width': width }"
		/>
	</div>
</template>
