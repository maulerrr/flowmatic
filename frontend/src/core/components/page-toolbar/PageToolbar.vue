<template>
	<div class="top-0 z-50 sticky bg-white/70 backdrop-blur-xl px-6 border-slate-200/50 border-b">
		<div class="mx-auto py-6">
			<div class="flex justify-between items-center gap-4">
				<div class="flex items-center space-x-4">
					<div class="space-y-1 max-w-3xl">
						<h1
							class="bg-clip-text bg-gradient-to-r from-slate-900 via-slate-800 to-slate-700 font-bold text-transparent text-xl lg:text-3xl line-clamp-2"
						>
							<slot name="title">{{ title }}</slot>
						</h1>
						<p class="text-slate-600 line-clamp-1">
							<slot name="description">{{ description }}</slot>
						</p>
					</div>
				</div>

				<div
					v-if="$slots.actions"
					class="flex items-center space-x-3"
				>
					<slot name="actions"></slot>
				</div>
			</div>
		</div>

		<!-- Subtle gradient line -->
		<div
			class="right-0 bottom-0 left-0 absolute bg-gradient-to-r from-transparent via-slate-200 to-transparent h-px"
		></div>
	</div>
</template>

<script setup lang="ts">
import { useSlots } from 'vue';





defineProps({
  title: {
    type: String,
    default: '',
  },
  description: {
    type: String,
    default: '',
  },
});

const $slots = useSlots();
</script>
