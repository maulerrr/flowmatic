<script setup lang="ts">
import { GraduationCap } from 'lucide-vue-next';
import { computed } from 'vue';



import { Sidebar, SidebarContent, SidebarHeader, type SidebarProps, SidebarRail, useSidebar } from '@/core/components/ui/sidebar';
import PrimaryNavMain from '@/core/layouts/primary/components/primary-nav-main.vue';
import { SIDEBAR_GROUPS } from '@/core/router/sidebar-menu.router';





// Import inject and computed

const props = withDefaults(defineProps<SidebarProps>(), {
	collapsible: 'icon',
})

// Inject the sidebar context to get its state
// This assumes the Shadcn Sidebar provides an injection key for its state.
// You might need to check Shadcn's specific documentation for the exact key.
// A common pattern is to inject a reactive object from the parent Sidebar component.
const {state} = useSidebar()

// Create a computed property to react to the collapsed state
const isSidebarCollapsed = computed(() => {
    // Access the collapsed state from the injected context
    // The exact property name might vary based on Shadcn's implementation.
    // Common names: 'collapsed', 'isCollapsed', 'state.collapsed'
    return state.value === 'collapsed';
});
</script>

<template>
	<Sidebar
		v-bind="props"
		class="bg-white border-gray-200 border-r"
	>
		<SidebarHeader class="bg-white border-gray-100 border-b">
			<div class="flex items-center gap-3">
				<div class="relative">
					<div
						class="flex justify-center items-center bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 shadow-lg rounded-xl size-9"
					>
						<GraduationCap class="size-6 text-white" />
					</div>
					<div
						class="-z-10 absolute inset-0 bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 opacity-30 blur-sm rounded-xl size-9"
					></div>
				</div>

				<Transition name="fade-slide">
					<div
						v-if="!isSidebarCollapsed"
						class="flex flex-col"
					>
						<h1
							class="bg-clip-text bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 font-bold text-transparent text-xl"
						>
							Flowmatic
						</h1>
						<span class="font-medium text-gray-500 text-xs tracking-wide"> Data Prep </span>
					</div>
				</Transition>
			</div>
		</SidebarHeader>

		<SidebarContent class="bg-white">
			<template
				v-for="group in SIDEBAR_GROUPS"
				:key="group.label"
			>
				<PrimaryNavMain
					:label="group.label"
					:items="group.items"
				/>
			</template>
		</SidebarContent>

		<SidebarRail />
	</Sidebar>
</template>

<style scoped>
/* Additional styling for enhanced visual appeal */
.sidebar-header-enhanced {
	background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

/* Custom hover effects for the logo */
.logo-container:hover .logo-icon {
	transform: scale(1.05);
	transition: transform 0.2s ease-in-out;
}

/* Smooth transitions */
* {
	transition: all 0.2s ease-in-out;
}

/* Transition for text fading/sliding */
.fade-slide-enter-active,
.fade-slide-leave-active {
	transition: all 0.2s ease-in-out;
	overflow: hidden; /* Important for width transitions */
}

.fade-slide-enter-from,
.fade-slide-leave-to {
	opacity: 0;
	transform: translateX(-10px); /* Optional: slight slide effect */
	width: 0; /* Collapse width */
	margin-left: 0; /* Remove margin */
	padding-left: 0; /* Remove padding */
	padding-right: 0; /* Remove padding */
}
</style>
