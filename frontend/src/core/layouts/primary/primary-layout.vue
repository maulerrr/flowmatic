<script lang="ts">




















export const description
  = 'A sidebar that collapses to icons.'
export const iframeHeight = '800px'
export const containerClass = 'w-full h-full'
</script>

<script setup lang="ts">
import { RouterView } from 'vue-router';



import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/core/components/ui/sidebar';
import { Skeleton } from "@/core/components/ui/skeleton";
import PrimaryNavUser from '@/core/layouts/primary/components/primary-nav-user.vue';
import PrimarySidebar from '@/core/layouts/primary/components/primary-sidebar.vue';



import { useCurrentUser } from "@/modules/auth/composables/current-user.composable";





const {data: currentUser} = useCurrentUser()
</script>

<template>
	<SidebarProvider>
		<PrimarySidebar />
		<SidebarInset>
			<header
				class="flex justify-between items-center gap-4 px-6 w-full h-16 group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12 transition-[width,height] ease-linear shrink-0"
			>
				<div class="flex items-center gap-2">
					<SidebarTrigger class="-ml-1" />
				</div>

				<div
					v-if="!currentUser"
					class="p-2"
				>
					<Skeleton class="w-[180px] h-full rounded-xl" />
				</div>

				<PrimaryNavUser
					v-else
					:user="currentUser"
				/>
			</header>
			<main class="bg-slate-50 min-h-screen">
				<RouterView />
			</main>
		</SidebarInset>
	</SidebarProvider>
</template>
