<script setup lang="ts">
import { PanelLeft } from 'lucide-vue-next';
import type { HTMLAttributes } from 'vue';



import { Button } from '@/core/components/ui/button';
import { cn } from '@/core/utils/tailwind.utils';



import { useSidebar } from './utils';





const props = defineProps<{
  class?: HTMLAttributes['class']
}>()

const { toggleSidebar } = useSidebar()
</script>

<template>
	<Button
		data-sidebar="trigger"
		data-slot="sidebar-trigger"
		variant="ghost"
		size="icon"
		:class="cn('h-7 w-7', props.class)"
		@click="toggleSidebar"
	>
		<PanelLeft />
		<span class="sr-only">Toggle Sidebar</span>
	</Button>
</template>
