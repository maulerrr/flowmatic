<script setup lang="ts">
import { reactiveOmit } from '@vueuse/core';
import { X } from 'lucide-vue-next';
import { DialogClose, DialogContent, type DialogContentEmits, type DialogContentProps, DialogPortal, useForwardPropsEmits } from 'reka-ui';
import type { HTMLAttributes } from 'vue';



import { cn } from '@/core/utils/tailwind.utils';



import DialogOverlay from './DialogOverlay.vue';





const { includeCloseButton = true, ...props} = defineProps<DialogContentProps & { class?: HTMLAttributes['class'], includeCloseButton?: boolean }>()
const emits = defineEmits<DialogContentEmits>()

const delegatedProps = reactiveOmit(props, 'class')

const forwarded = useForwardPropsEmits(delegatedProps, emits)
</script>

<template>
	<DialogPortal>
		<DialogOverlay />
		<DialogContent
			data-slot="dialog-content"
			v-bind="forwarded"
			:class="
        cn(
          'top-[50%] left-[50%] z-50 fixed gap-4 grid bg-background shadow-lg p-6 border rounded-lg w-full max-w-[calc(100%-2rem)] sm:max-w-lg translate-x-[-50%] translate-y-[-50%] data-[state=closed]:animate-out data-[state=open]:animate-in duration-200 data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
          props.class,
        )"
		>
			<slot />

			<DialogClose
				v-if="includeCloseButton"
				class="ring-offset-background focus:ring-ring data-[state=open]:bg-accent data-[state=open]:text-muted-foreground absolute top-4 right-4 rounded-xs opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
			>
				<X />
				<span class="sr-only">Close</span>
			</DialogClose>
		</DialogContent>
	</DialogPortal>
</template>
