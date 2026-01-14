<script lang="ts">































export const description = 'Flowmatic login — data orchestration portal'
</script>

<script setup lang="ts">
import { ArrowLeft, Database, Sparkles } from 'lucide-vue-next';



import LoginForm from '@/modules/auth/components/login-form.vue';
</script>

<template>
	<div
		class="min-h-screen bg-gradient-to-br from-surface-1 via-surface-2 to-surface-1 text-foreground relative overflow-hidden"
	>
		<div
			class="absolute inset-0 opacity-60"
			style="background: radial-gradient(circle at 20% 20%, rgba(52, 208, 195, 0.14), transparent 32%), radial-gradient(circle at 80% 10%, rgba(124, 140, 255, 0.18), transparent 30%), radial-gradient(circle at 50% 80%, rgba(240, 93, 94, 0.12), transparent 36%);"
		></div>
		<div
			class="relative max-w-6xl mx-auto px-4 lg:px-10 py-12 lg:py-20 grid gap-10 lg:grid-cols-[1.1fr_0.9fr] items-center"
		>
			<div class="space-y-6">
				<div
					class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/15 text-primary text-xs font-semibold uppercase tracking-[0.18em]"
				>
					<Sparkles class="w-4 h-4" />
					Flowmatic
				</div>
				<h1 class="text-4xl lg:text-5xl font-bold leading-tight">
					Welcome back to your data flow cockpit.
				</h1>
				<p class="text-foreground/70 text-base max-w-2xl">
					Authenticate to orchestrate ingestion, quality, and exports with a single continuous
					surface. Your last session resumes instantly.
				</p>
				<div class="grid grid-cols-2 gap-4">
					<div class="rounded-2xl border border-border bg-surface-2 p-4 shadow-sm">
						<p class="text-xs uppercase tracking-[0.16em] text-foreground/60">
							Pipelines monitored
						</p>
						<p class="text-2xl font-bold">24</p>
					</div>
					<div class="rounded-2xl border border-border bg-surface-2 p-4 shadow-sm">
						<p class="text-xs uppercase tracking-[0.16em] text-foreground/60">Avg. clean rate</p>
						<p class="text-2xl font-bold text-success">98%</p>
					</div>
				</div>
				<router-link
					to="/"
					class="inline-flex items-center gap-2 text-foreground/70 hover:text-foreground font-semibold transition"
				>
					<ArrowLeft class="w-4 h-4" />
					Back to site
				</router-link>
			</div>

			<div class="relative">
				<div class="absolute inset-0 blur-3xl bg-primary/10 rounded-3xl" />
				<div
					class="relative rounded-3xl border border-border bg-surface-2/90 backdrop-blur-xl shadow-2xl p-8 lg:p-10"
				>
					<div class="flex items-center gap-3 mb-6">
						<div
							class="w-11 h-11 rounded-2xl bg-[var(--gradient-accent)] flex items-center justify-center shadow-[var(--glow)]"
						>
							<Database class="w-6 h-6 text-sidebar-foreground" />
						</div>
						<div>
							<p class="text-sm uppercase tracking-[0.18em] text-foreground/60">Flowmatic</p>
							<p class="text-lg font-semibold">Sign in</p>
						</div>
					</div>

					<div class="space-y-6">
						<LoginForm />
						<p class="text-xs text-foreground/60 text-center">
							By signing in you accept the platform terms and privacy notice.
						</p>
					</div>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped>
@keyframes pulse {
  0%, 100% {
    opacity: 0.4;
  }
  50% {
    opacity: 0.6;
  }
}

.animate-pulse {
  animation: pulse 4s ease-in-out infinite;
}

.delay-500 {
  animation-delay: 0.5s;
}

.delay-1000 {
  animation-delay: 1s;
}
</style>
