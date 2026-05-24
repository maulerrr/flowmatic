<script setup lang="ts">
import { apiClient } from '@/api/client';
import { AlertCircle, LogIn, Mail, Sparkles, UserPlus } from 'lucide-vue-next';
import { ref } from 'vue';
import { useRouter } from 'vue-router';





const router = useRouter()
const mode = ref<'login' | 'register'>('login')
const email = ref('')
const password = ref('')
const displayName = ref('')
const organizationName = ref('')
const isLoading = ref(false)
const error = ref('')

async function handleSubmit() {
	if (!email.value || !password.value) {
		error.value = 'Please enter your email and password'
		return
	}
	if (mode.value === 'register' && !displayName.value) {
		error.value = 'Please enter your name'
		return
	}

	try {
		isLoading.value = true
		error.value = ''

		if (mode.value === 'login') {
			await apiClient.login(email.value, password.value)
		} else {
			await apiClient.register({
				email: email.value,
				password: password.value,
				displayName: displayName.value,
				organizationName: organizationName.value || undefined,
			})
		}

		router.push('/dashboard')
	} catch (err: any) {
		error.value = err.message || 'Login failed. Please try again.'
	} finally {
		isLoading.value = false
	}
}

function toggleMode() {
	mode.value = mode.value === 'login' ? 'register' : 'login'
	error.value = ''
}
</script>

<template>
	<div
		class="min-h-screen bg-background flex flex-col items-center justify-center relative overflow-hidden p-4"
	>
		<!-- Animated background elements -->
		<div class="absolute inset-0 overflow-hidden pointer-events-none">
			<div
				class="absolute -top-40 -right-40 w-80 h-80 bg-primary/10 rounded-full blur-3xl opacity-30 animate-pulse"
			></div>
			<div
				class="absolute -bottom-40 -left-40 w-80 h-80 bg-secondary/10 rounded-full blur-3xl opacity-30 animate-pulse"
				style="animation-delay: 1s;"
			></div>
		</div>

		<!-- Main Content -->
		<div class="w-full max-w-md relative z-10">
			<!-- Logo Section -->
			<div class="text-center mb-12 space-y-3">
				<div class="flex justify-center mb-6">
					<div
						class="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-[0_10px_40px_rgba(52,208,195,0.3)]"
					>
						<Sparkles class="w-8 h-8 text-foreground" />
						<div
							class="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary to-secondary opacity-20 blur-xl -z-10"
						></div>
					</div>
				</div>
				<div>
					<h1 class="text-3xl md:text-4xl font-bold text-foreground tracking-tight">Flowmatic</h1>
					<p class="text-foreground/60 mt-2 text-sm md:text-base">Intelligent Data Preparation</p>
				</div>
			</div>

			<!-- Login Card -->
			<div
				class="rounded-2xl border border-border bg-card/50 backdrop-blur-xl shadow-2xl overflow-hidden"
			>
				<div class="p-8 md:p-10 space-y-8">
					<!-- Heading -->
					<div>
						<h2 class="text-xl md:text-2xl font-bold text-foreground">
							{{ mode === 'login' ? 'Welcome back' : 'Create your workspace' }}
						</h2>
						<p class="text-foreground/60 text-sm mt-1">
							{{ mode === 'login' ? 'Sign in with your email and password' : 'Your first organization will be created automatically' }}
						</p>
					</div>

					<form
						@submit.prevent="handleSubmit"
						class="space-y-5"
					>
						<div
							v-if="mode === 'register'"
							class="space-y-2.5"
						>
							<label
								for="displayName"
								class="text-sm font-medium text-foreground/80"
								>Your Name</label
							>
							<input
								id="displayName"
								v-model="displayName"
								type="text"
								required
								placeholder="Ada Lovelace"
								:disabled="isLoading"
								class="w-full px-4 py-3 bg-input border border-border rounded-xl text-foreground placeholder:text-foreground/40 transition-all duration-200 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
							/>
						</div>
						<!-- Email Input -->
						<div class="space-y-2.5">
							<label
								for="email"
								class="text-sm font-medium text-foreground/80"
								>Email Address</label
							>
							<div class="relative">
								<Mail class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground/40" />
								<input
									id="email"
									v-model="email"
									type="email"
									required
									placeholder="you@example.com"
									:disabled="isLoading"
									class="w-full pl-10 pr-4 py-3 bg-input border border-border rounded-xl text-foreground placeholder:text-foreground/40 transition-all duration-200 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
								/>
							</div>
						</div>

						<div class="space-y-2.5">
							<label
								for="password"
								class="text-sm font-medium text-foreground/80"
								>Password</label
							>
							<input
								id="password"
								v-model="password"
								type="password"
								required
								minlength="6"
								placeholder="At least 6 characters"
								:disabled="isLoading"
								class="w-full px-4 py-3 bg-input border border-border rounded-xl text-foreground placeholder:text-foreground/40 transition-all duration-200 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
							/>
						</div>

						<div
							v-if="mode === 'register'"
							class="space-y-2.5"
						>
							<label
								for="organizationName"
								class="text-sm font-medium text-foreground/80"
								>Organization Name</label
							>
							<input
								id="organizationName"
								v-model="organizationName"
								type="text"
								placeholder='Defaults to "Your Name&apos;s Organization"'
								:disabled="isLoading"
								class="w-full px-4 py-3 bg-input border border-border rounded-xl text-foreground placeholder:text-foreground/40 transition-all duration-200 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
							/>
						</div>

						<!-- Error Message -->
						<transition name="slide-fade">
							<div
								v-if="error"
								class="flex items-start gap-3 bg-destructive/10 border border-destructive/30 rounded-xl p-4"
							>
								<AlertCircle class="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
								<p class="text-destructive text-sm font-medium">{{ error }}</p>
							</div>
						</transition>

						<!-- Submit Button -->
						<button
							type="submit"
							:disabled="isLoading"
							class="w-full py-3 px-4 rounded-xl font-semibold transition-all duration-200 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed"
							:class="isLoading ? 'bg-primary/70' : 'bg-gradient-to-r from-primary to-secondary hover:shadow-[var(--glow)] text-foreground'"
						>
							<span
								v-if="!isLoading"
								class="flex items-center justify-center gap-2"
							>
								<span>Sign In</span>
								<component
									:is="mode === 'login' ? LogIn : UserPlus"
									class="w-4 h-4"
								/>
							</span>
							<span
								v-else
								class="flex items-center justify-center gap-2"
							>
								<svg
									class="animate-spin w-4 h-4"
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
								>
									<circle
										class="opacity-25"
										cx="12"
										cy="12"
										r="10"
										stroke="currentColor"
										stroke-width="4"
									></circle>
									<path
										class="opacity-75"
										fill="currentColor"
										d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
									></path>
								</svg>
								<span>{{ mode === 'login' ? 'Signing in...' : 'Creating account...' }}</span>
							</span>
						</button>
					</form>

					<!-- Divider -->
					<div class="relative py-2">
						<div class="absolute inset-0 flex items-center">
							<div class="w-full border-t border-border/50"></div>
						</div>
						<div class="relative flex justify-center text-xs">
							<span class="px-3 bg-card text-foreground/50">Demo Credentials</span>
						</div>
					</div>

					<button
						type="button"
						@click="toggleMode"
						class="w-full text-sm text-primary hover:text-primary/80 font-semibold"
					>
						{{ mode === 'login' ? 'Need an account? Register' : 'Already have an account? Sign in' }}
					</button>
				</div>

				<!-- Footer -->
				<div class="px-8 py-6 md:py-8 border-t border-border/50 bg-card/30 backdrop-blur-sm">
					<p class="text-xs text-foreground/50 text-center leading-relaxed">
						By signing in, you agree to our
						<a
							href="#"
							class="text-primary hover:text-primary/80 underline"
							>Terms of Service</a
						>
						and
						<a
							href="#"
							class="text-primary hover:text-primary/80 underline"
							>Privacy Policy</a
						>
					</p>
				</div>
			</div>

			<!-- Footer Text -->
			<div class="mt-8 text-center space-y-2">
				<p class="text-sm text-foreground/60">
					Built with <span class="text-primary">❤</span> for data professionals
				</p>
			</div>
		</div>
	</div>
</template>
