<script setup lang="ts">
import 'vue-sonner/style.css'
import { Toaster } from '@/core/components/ui/sonner'
import SidebarLayout from '@/layouts/sidebar-layout.vue'
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
