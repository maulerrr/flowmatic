<script setup lang="ts">
import SidebarLayout from '@/layouts/sidebar-layout.vue';
import { computed } from 'vue';
import { RouterView, useRoute } from 'vue-router';
import 'vue-sonner/style.css';



import { Toaster } from '@/core/components/ui/sonner';
import { isDark } from '@/core/composables/theme';





// Initialize theme
isDark.value

const route = useRoute()
const usesPublicLayout = computed(() => route.meta.requiresAuth === false)
</script>

<template>
	<Toaster
		class="pointer-events-auto"
		:duration="3000"
		rich-colors
	/>
	<RouterView v-if="usesPublicLayout" />
	<SidebarLayout v-else />
</template>

<style scoped></style>
