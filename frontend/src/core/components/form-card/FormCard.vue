<template>
	<Card class="bg-white shadow-sm border border-slate-200/60 rounded-xl">
		<CardHeader class="pb-4">
			<CardTitle class="font-bold text-slate-900">
				<slot name="title"></slot>
			</CardTitle>
			<CardDescription class="text-slate-600">
				<slot name="description"></slot>
			</CardDescription>
		</CardHeader>
		<CardContent class="gap-4 grid bg-white"> <slot></slot> </CardContent>
	</Card>
</template>

<script setup lang="ts">
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/core/components/ui/card';





// No props or emits needed, purely a structural component
</script>

<style scoped>
/* No specific styles needed here */
</style>
