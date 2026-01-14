<template>
	<div
		class="top-0 z-40 sticky bg-white/80 dark:bg-gray-900/70 shadow-sm backdrop-blur-lg px-6 py-4 border-gray-100 dark:border-gray-800 border-b md:rounded-xl rounded-b-xl"
	>
		<div class="flex items-center gap-4">
			<Button
				variant="outline"
				size="icon"
				@click="$emit('cancel')"
				class="hover:bg-gray-100 dark:hover:bg-gray-700 w-10 h-10 text-gray-700 dark:text-gray-200 transition-all duration-200"
			>
				<ChevronLeft class="w-5 h-5" />
				<span class="sr-only">Артқа</span>
			</Button>
			<h1
				class="flex-1 font-semibold text-gray-900 dark:text-gray-100 text-xl leading-tight tracking-tight whitespace-nowrap sm:grow-0 shrink-0"
			>
				<slot name="title">Form Title</slot>
			</h1>

			<div class="hidden md:flex items-center gap-3 md:ml-auto">
				<slot name="additionalActions"></slot>

				<Button
					type="button"
					variant="outline"
					@click="$emit('cancel')"
				>
					<slot name="cancel-button-text">Артқа</slot>
				</Button>
				<Button
					type="submit"
					:disabled="isSubmitting"
					variant="gradient"
					@click="$emit('save')"
				>
					<Loader2
						v-if="isSubmitting"
						class="mr-2 w-4 h-4 animate-spin"
					/>
					<slot name="save-button-text">Сақтау</slot>
				</Button>
			</div>
		</div>
	</div>
</template>

<script setup lang="ts">
import { ChevronLeft, Loader2 } from 'lucide-vue-next';



import { Button } from '@/core/components/ui/button';





// Added Loader2 for loading state

defineProps({
  isSubmitting: {
    type: Boolean,
    default: false,
  },
});

defineEmits(['cancel', 'save']);
</script>

<style scoped>
/* No specific styles needed here as Tailwind classes are inline */
</style>
