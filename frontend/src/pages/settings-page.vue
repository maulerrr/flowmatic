<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Bell, Lock, Palette, Database, X } from 'lucide-vue-next'
import { apiClient, type User } from '@/api/client'
import { toast } from 'vue-sonner' // Assuming toast is available or I'll use console/alert if not

const settings = [
	{
		icon: Bell,
		title: 'Notifications',
		description: 'Manage your notification preferences',
		color: 'from-blue-500 to-cyan-500',
	},
	{
		icon: Lock,
		title: 'Security',
		description: 'Password and authentication settings',
		color: 'from-green-500 to-emerald-500',
	},
	{
		icon: Palette,
		title: 'Appearance',
		description: 'Customize your workspace look and feel',
		color: 'from-purple-500 to-pink-500',
	},
	{
		icon: Database,
		title: 'Data & Privacy',
		description: 'Manage data retention and privacy',
		color: 'from-orange-500 to-red-500',
	},
]

const user = ref<User | null>(null)
const loading = ref(false)

// Theme State
const isDark = ref(false)

// Password State
const showPasswordModal = ref(false)
const newPassword = ref('')
const confirmPassword = ref('')
const passwordLoading = ref(false)

// Delete Account State
const showDeleteModal = ref(false)
const deleteLoading = ref(false)

onMounted(async () => {
	// Theme Init
	const savedTheme = localStorage.getItem('theme')
	if (savedTheme) {
		isDark.value = savedTheme === 'dark'
	} else {
		isDark.value = window.matchMedia('(prefers-color-scheme: dark)').matches
	}
	updateTheme()

	loading.value = true
	try {
		const response = await apiClient.getProfile()
		if (response.success && response.data) {
			user.value = response.data
		}
	} catch (error) {
		console.error('Failed to fetch user profile', error)
	} finally {
		loading.value = false
	}
})

const updateTheme = () => {
	const el = document.documentElement
	if (isDark.value) {
		el.classList.add('dark')
		localStorage.setItem('theme', 'dark')
	} else {
		el.classList.remove('dark')
		localStorage.setItem('theme', 'light')
	}
}

const toggleTheme = () => {
	isDark.value = !isDark.value
	updateTheme()
}

const handleChangePassword = async () => {
	if (newPassword.value.length < 6) {
		toast.error('Password must be at least 6 characters')
		return
	}
	if (newPassword.value !== confirmPassword.value) {
		toast.error('Passwords do not match')
		return
	}

	passwordLoading.value = true
	try {
		const res = await apiClient.changePassword(newPassword.value)
		if (res.success) {
			toast.success('Password updated successfully')
			showPasswordModal.value = false
			newPassword.value = ''
			confirmPassword.value = ''
		} else {
			toast.error('Failed to update password')
		}
	} catch (e) {
		toast.error('An error occurred')
	} finally {
		passwordLoading.value = false
	}
}

const handleDeleteAccount = async () => {
	deleteLoading.value = true
	try {
		const res = await apiClient.deleteAccount()
		if (res.success) {
			toast.success('Account deleted. Goodbye.')
			window.location.href = '/login'
		} else {
			toast.error('Failed to delete account')
		}
	} catch (e) {
		toast.error('An error occurred')
	} finally {
		deleteLoading.value = false
	}
}

const formatDate = (dateString?: string) => {
	if (!dateString) return 'Unknown'
	return new Date(dateString).toLocaleDateString('en-US', {
		year: 'numeric',
		month: 'long',
		day: 'numeric'
	})
}
</script>

<template>
	<div class="min-h-screen bg-background">
		<!-- Header -->
		<div class="px-6 py-8 md:px-8 border-b border-border/50">
			<div class="max-w-7xl mx-auto">
				<h1 class="text-3xl md:text-4xl font-bold text-foreground mb-2">Settings</h1>
				<p class="text-foreground/60">Manage your account and application preferences</p>
			</div>
		</div>

		<!-- Content -->
		<div class="max-w-3xl mx-auto px-6 py-8 md:px-8">
			<!-- Settings Sections -->
			<div class="space-y-6 mb-12">
				<div v-for="setting in settings" :key="setting.title" class="rounded-xl border border-border bg-card/70 backdrop-blur-md overflow-hidden hover:border-primary/40 transition-all group">
					<div class="p-6 flex items-start justify-between gap-4">
						<div class="flex items-start gap-4 flex-1">
							<div :class="['w-12 h-12 rounded-lg bg-gradient-to-br flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform', setting.color]">
								<component :is="setting.icon" class="w-6 h-6 text-white" />
							</div>
							<div>
								<h3 class="text-lg font-bold text-foreground mb-1">{{ setting.title }}</h3>
								<p class="text-foreground/60 text-sm">{{ setting.description }}</p>
							</div>
						</div>
						<button class="px-4 py-2 rounded-lg border border-border hover:border-primary/40 text-foreground/80 hover:text-foreground font-medium text-sm transition whitespace-nowrap">
							Configure
						</button>
					</div>
				</div>
			</div>

			<!-- Account Section -->
			<div class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-6 mb-8">
				<h3 class="text-lg font-bold text-foreground mb-4">Account Information</h3>
				<div v-if="loading" class="text-sm text-foreground/60">Loading profile...</div>
				<div v-else class="space-y-4">
					<div>
						<label class="block text-sm font-medium text-foreground mb-2">Email Address</label>
						<input type="email" :value="user?.email" disabled class="w-full px-4 py-2 rounded-lg border border-border bg-border/20 text-foreground/70 disabled:cursor-not-allowed" />
					</div>
					<div>
						<label class="block text-sm font-medium text-foreground mb-2">Account Role</label>
						<input type="text" :value="user?.role || 'User'" disabled class="w-full px-4 py-2 rounded-lg border border-border bg-border/20 text-foreground/70 disabled:cursor-not-allowed" />
					</div>
					<div class="pt-4 border-t border-border/30">
						<p class="text-xs text-foreground/50 mb-3">Member since {{ formatDate(user?.createdAt) }}</p>
						<button @click="showPasswordModal = true" class="px-4 py-2 rounded-lg border border-border hover:border-primary/40 text-foreground/80 hover:text-foreground font-medium text-sm transition">
							Change Password
						</button>
					</div>
				</div>
			</div>

			<!-- Preferences Section -->
			<div class="rounded-xl border border-border bg-card/70 backdrop-blur-md p-6 mb-8">
				<h3 class="text-lg font-bold text-foreground mb-4">Preferences</h3>
				<div class="space-y-4">
					<div class="flex items-center justify-between">
						<div>
							<p class="font-medium text-foreground">Email Notifications</p>
							<p class="text-sm text-foreground/60">Receive updates about your pipelines</p>
						</div>
						<label class="relative inline-flex cursor-pointer">
							<input type="checkbox" checked class="sr-only peer" />
							<div class="w-11 h-6 bg-border rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
						</label>
					</div>
					<div class="flex items-center justify-between pt-4 border-t border-border/20">
						<div>
							<p class="font-medium text-foreground">Dark Mode</p>
							<p class="text-sm text-foreground/60">Toggle workspace theme</p>
						</div>
						<label class="relative inline-flex cursor-pointer">
							<input type="checkbox" :checked="isDark" @change="toggleTheme" class="sr-only peer" />
							<div class="w-11 h-6 bg-border rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
						</label>
					</div>
				</div>
			</div>

			<!-- Danger Zone -->
			<div class="rounded-xl border border-destructive/30 bg-destructive/10 backdrop-blur-md p-6">
				<h3 class="text-lg font-bold text-destructive mb-2">Danger Zone</h3>
				<p class="text-foreground/60 text-sm mb-4">Once you delete your account, there is no going back. Please be certain.</p>
				<button @click="showDeleteModal = true" class="px-6 py-2 rounded-lg bg-destructive/15 border border-destructive/30 hover:bg-destructive/25 text-destructive font-medium text-sm transition">
					Delete Account
				</button>
			</div>
		</div>

		<!-- Change Password Modal -->
		<div v-if="showPasswordModal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
			<div class="bg-card border border-border rounded-xl w-full max-w-md p-6 shadow-xl relative">
				<div class="flex items-center justify-between mb-6">
					<h3 class="text-xl font-bold text-foreground">Change Password</h3>
					<button @click="showPasswordModal = false" class="text-foreground/60 hover:text-foreground">
						<X class="w-5 h-5" />
					</button>
				</div>
				<div class="space-y-4">
					<div>
						<label class="block text-sm font-medium text-foreground mb-1">New Password</label>
						<input v-model="newPassword" type="password" placeholder="Min. 6 characters" class="w-full px-4 py-2 rounded-lg border border-border bg-input text-foreground focus:ring-2 focus:ring-primary/50" />
					</div>
					<div>
						<label class="block text-sm font-medium text-foreground mb-1">Confirm Password</label>
						<input v-model="confirmPassword" type="password" placeholder="Re-enter password" class="w-full px-4 py-2 rounded-lg border border-border bg-input text-foreground focus:ring-2 focus:ring-primary/50" />
					</div>
				</div>
				<div class="mt-6 flex justify-end gap-3">
					<button @click="showPasswordModal = false" class="px-4 py-2 text-sm text-foreground/70 hover:text-foreground">Cancel</button>
					<button 
						@click="handleChangePassword" 
						:disabled="passwordLoading"
						class="px-4 py-2 bg-primary text-primary-foreground font-medium rounded-lg text-sm disabled:opacity-50"
					>
						{{ passwordLoading ? 'Updating...' : 'Update Password' }}
					</button>
				</div>
			</div>
		</div>

		<!-- Delete Account Modal -->
		<div v-if="showDeleteModal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
			<div class="bg-card border border-destructive/50 rounded-xl w-full max-w-md p-6 shadow-xl relative">
				<div class="flex items-center justify-between mb-4">
					<h3 class="text-xl font-bold text-destructive">Delete Account?</h3>
					<button @click="showDeleteModal = false" class="text-foreground/60 hover:text-foreground">
						<X class="w-5 h-5" />
					</button>
				</div>
				<p class="text-foreground/80 mb-6">
					This action cannot be undone. This will permanently delete your account, organization, and all associated data.
				</p>
				<div class="flex justify-end gap-3">
					<button @click="showDeleteModal = false" class="px-4 py-2 text-sm text-foreground/70 hover:text-foreground">Cancel</button>
					<button 
						@click="handleDeleteAccount" 
						:disabled="deleteLoading"
						class="px-4 py-2 bg-destructive text-destructive-foreground font-medium rounded-lg text-sm disabled:opacity-50"
					>
						{{ deleteLoading ? 'Deleting...' : 'Yes, Delete Everything' }}
					</button>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped></style>
