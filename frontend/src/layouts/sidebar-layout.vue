<script setup lang="ts">
import { apiClient } from '@/api/client';
import type { User as UserType } from '@/api/client';
import { Activity, BarChart3, Bell, ChevronLeft, ChevronRight, Database, LogOut, Menu, MessageSquareText, Network, Settings, Sparkles, Upload, User, X } from 'lucide-vue-next';
import { onMounted, ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';





const router = useRouter()
const route = useRoute()
const sidebarOpen = ref(false)
const sidebarCollapsed = ref(false)
const user = ref<UserType | null>(null)
const userName = ref('Loading...')
const userEmail = ref('')
const isLoggingOut = ref(false)

const menuItems = [
	{ icon: Database, label: 'Dashboard', path: '/dashboard', badge: null },
	{ icon: Network, label: 'Pipeline Workbench', path: '/connectors', badge: 'live' },
	{ icon: MessageSquareText, label: 'Pipeline Insights', path: '/insights', badge: null },
	{ icon: Upload, label: 'Upload', path: '/upload', badge: null },
	{ icon: Activity, label: 'Pipelines', path: '/pipelines', badge: 'live' },
	{ icon: BarChart3, label: 'Analytics', path: '/analytics', badge: null },
	{ icon: Settings, label: 'Settings', path: '/settings', badge: null },
]

const isActive = (path: string) => route.path === path

onMounted(async () => {
	try {
		const res = await apiClient.getProfile()
		if (res.success && res.data) {
			user.value = res.data
			userName.value = res.data.displayName
			userEmail.value = res.data.email
		}
	} catch (error) {
		console.error('Failed to load user profile:', error)
	}
})

async function handleLogout() {
	try {
		isLoggingOut.value = true
		await apiClient.logout()
	} catch (error) {
		console.error('Logout failed:', error)
	} finally {
		user.value = null
		userName.value = ''
		userEmail.value = ''
		isLoggingOut.value = false
		router.push('/login')
	}
}
</script>

<template>
	<div class="min-h-screen text-foreground flex bg-background">
		<!-- Sidebar -->
		<aside
			:class="[
				'fixed inset-y-0 left-0 z-50 w-72 bg-sidebar/95 backdrop-blur-xl border-r border-sidebar-border transform transition-all duration-300 lg:relative lg:translate-x-0 flex flex-col',
				sidebarOpen ? 'translate-x-0' : '-translate-x-full',
				sidebarCollapsed ? 'lg:w-24' : 'lg:w-72',
			]"
		>
			<!-- Logo -->
			<div class="p-6 border-b border-sidebar-border flex items-center gap-3 flex-shrink-0">
				<div
					class="w-11 h-11 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0 shadow-[var(--glow)]"
				>
					<Sparkles class="w-6 h-6 text-white" />
				</div>
				<div
					v-if="!sidebarCollapsed"
					class="transition-all min-w-0"
				>
					<h1 class="text-lg font-bold tracking-tight truncate">Flowmatic</h1>
					<p class="text-xs text-sidebar-foreground/70 truncate">Data Prep</p>
				</div>
			</div>

			<!-- Navigation -->
			<nav class="flex-1 p-4 space-y-2 overflow-y-auto">
				<router-link
					v-for="item in menuItems"
					:key="item.path"
					:to="item.path"
					:class="[
						'flex items-center gap-3 px-4 py-3 rounded-xl transition-all group relative overflow-hidden',
						isActive(item.path)
							? 'bg-gradient-to-r from-primary/20 to-secondary/10 text-primary border border-primary/40 shadow-[var(--glow)]'
							: 'text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-border/50',
					]"
				>
					<component
						:is="item.icon"
						class="w-5 h-5 flex-shrink-0 group-hover:scale-110 transition-transform"
					/>
					<span
						v-if="!sidebarCollapsed"
						class="flex-1 font-medium text-sm"
						>{{ item.label }}</span
					>
					<span
						v-if="item.badge && !sidebarCollapsed"
						class="px-2 py-0.5 bg-primary/15 text-primary text-xs rounded-full font-semibold"
					>
						{{ item.badge }}
					</span>
					<ChevronRight
						v-if="isActive(item.path) && !sidebarCollapsed"
						class="w-4 h-4 absolute right-3 opacity-0 group-hover:opacity-100 transition"
					/>
				</router-link>
			</nav>

			<!-- User Profile Section -->
			<div class="p-4 border-t border-sidebar-border space-y-3 flex-shrink-0">
				<div
					v-if="user"
					class="flex items-center gap-3 p-3 rounded-xl bg-sidebar-border/40 hover:bg-sidebar-border/60 transition"
				>
					<div
						class="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0"
					>
						<User class="w-5 h-5 text-white" />
					</div>
					<div
						v-if="!sidebarCollapsed"
						class="flex-1 min-w-0"
					>
						<p class="text-sm font-semibold truncate">{{ userName }}</p>
						<p class="text-xs text-sidebar-foreground/60 truncate">{{ userEmail }}</p>
					</div>
				</div>
				<div
					v-else
					class="flex items-center gap-3 p-3 rounded-xl bg-sidebar-border/40 animate-pulse"
				>
					<div class="w-10 h-10 rounded-full bg-sidebar-border/60"></div>
					<div
						v-if="!sidebarCollapsed"
						class="flex-1 space-y-2"
					>
						<div class="h-3 bg-sidebar-border/60 rounded w-24"></div>
						<div class="h-2 bg-sidebar-border/60 rounded w-32"></div>
					</div>
				</div>
				<button
					@click="handleLogout"
					:disabled="isLoggingOut"
					class="w-full flex items-center justify-center gap-2 px-4 py-2 text-destructive hover:bg-destructive/15 rounded-xl transition text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
				>
					<LogOut class="w-4 h-4" />
					<span v-if="!sidebarCollapsed">{{ isLoggingOut ? 'Logging out...' : 'Logout' }}</span>
				</button>
			</div>
		</aside>

		<!-- Main Content -->
		<div class="flex-1 flex flex-col">
			<!-- Top Bar -->
			<header class="sticky top-0 z-40 bg-card/40 backdrop-blur-xl border-b border-border/50">
				<div class="flex items-center justify-between px-6 py-4">
					<div class="flex items-center gap-4">
						<button
							@click="sidebarOpen = !sidebarOpen"
							class="lg:hidden p-2 hover:bg-border/50 rounded-xl transition text-foreground/70 hover:text-foreground"
						>
							<Menu
								v-if="!sidebarOpen"
								class="w-6 h-6"
							/>
							<X
								v-else
								class="w-6 h-6"
							/>
						</button>
						<button
							@click="sidebarCollapsed = !sidebarCollapsed"
							class="hidden lg:inline-flex p-2 hover:bg-border/50 rounded-xl transition text-foreground/70 hover:text-foreground"
							:title="sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
						>
							<ChevronRight
								v-if="sidebarCollapsed"
								class="w-5 h-5"
							/>
							<ChevronLeft
								v-else
								class="w-5 h-5"
							/>
						</button>
					</div>

					<!-- Right Actions -->
					<div class="flex items-center gap-3">
						<!-- Notifications hidden for now -->
					</div>
				</div>
			</header>

			<!-- Content Area -->
			<main class="flex-1 overflow-auto">
				<RouterView />
			</main>
		</div>

		<!-- Mobile Sidebar Overlay -->
		<div
			v-if="sidebarOpen"
			@click="sidebarOpen = false"
			class="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
		/>
	</div>
</template>

<style scoped></style>
