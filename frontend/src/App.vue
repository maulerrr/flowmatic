<script setup lang="ts">
import SidebarLayout from '@/layouts/sidebar-layout.vue';
import 'vue-sonner/style.css';



import { Toaster } from '@/core/components/ui/sonner';
</script>

<template>
	<Toaster
		class="pointer-events-auto"
		:duration="3000"
		rich-colors
	/>
	<SidebarLayout />
</template>

<style scoped></style>
