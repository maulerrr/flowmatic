<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { Upload, CheckCircle, AlertCircle, Loader2, FileText, Database } from 'lucide-vue-next'
import { Button } from '@/core/components/ui/button'
import { apiClient } from '@/api/client'

const router = useRouter()
const isDragging = ref(false)
const selectedFile = ref<File | null>(null)
const isLoading = ref(false)
const uploadProgress = ref(0)
const uploadStatus = ref<'idle' | 'uploading' | 'success' | 'error'>('idle')
const errorMessage = ref('')
const currentRunId = ref<string | null>(null)

const fileInputRef = ref<HTMLInputElement>()

const fileName = computed(() => selectedFile.value?.name || '')
const fileSize = computed(() => {
	if (!selectedFile.value) return ''
	const kb = selectedFile.value.size / 1024
	return kb > 1024 ? `${(kb / 1024).toFixed(2)} MB` : `${kb.toFixed(2)} KB`
})

const handleDragover = (e: DragEvent) => {
	e.preventDefault()
	isDragging.value = true
}

const handleDragleave = () => {
	isDragging.value = false
}

const handleDrop = (e: DragEvent) => {
	e.preventDefault()
	isDragging.value = false
	const files = e.dataTransfer?.files
	if (files?.length) {
		selectedFile.value = files[0]
	}
}

const handleFileSelect = (e: Event) => {
	const input = e.target as HTMLInputElement
	if (input.files?.length) {
		selectedFile.value = input.files[0]
	}
}

const handleUpload = async () => {
	if (!selectedFile.value) return

	isLoading.value = true
	uploadStatus.value = 'uploading'
	errorMessage.value = ''
	uploadProgress.value = 0

	try {
		// Upload file and get run ID
		const response = await apiClient.uploadFile(selectedFile.value)

		if (!response.success || !response.data) {
			throw new Error(response.error || 'Upload failed')
		}

		const { runId } = response.data
		currentRunId.value = runId

		uploadProgress.value = 100
		uploadStatus.value = 'success'

		// Redirect to pipelines page to view the run
		setTimeout(() => {
			router.push(`/pipelines?runId=${runId}`)
		}, 2000)
	} catch (error) {
		uploadStatus.value = 'error'
		errorMessage.value = error instanceof Error ? error.message : 'Upload failed'
		isLoading.value = false
	}
}

const triggerFileInput = () => {
	fileInputRef.value?.click()
}

const resetUpload = () => {
	selectedFile.value = null
	uploadStatus.value = 'idle'
	uploadProgress.value = 0
	errorMessage.value = ''
	currentRunId.value = null
}
</script>

<template>
	<div class="min-h-screen bg-background">
		<!-- Header Section -->
		<div class="px-6 py-8 md:px-8 border-b border-border/50">
			<div class="max-w-5xl mx-auto">
				<div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/15 text-primary text-xs font-semibold uppercase tracking-[0.15em] mb-4">
					<Database class="w-4 h-4" />
					Data Ingestion
				</div>
				<h1 class="text-3xl md:text-4xl font-bold text-foreground mb-2">Upload Data</h1>
				<p class="text-foreground/60 max-w-2xl">
					Drop your CSV or JSON file to start the data preparation pipeline. Automatic quality checks and cleaning will begin immediately.
				</p>
			</div>
		</div>

		<!-- Main Content -->
		<div class="max-w-5xl mx-auto px-6 py-8 md:px-8">
			<!-- Upload Card - Idle State -->
			<div v-if="uploadStatus === 'idle'" class="space-y-6">
				<div class="rounded-2xl border border-border bg-card/70 backdrop-blur-md overflow-hidden">
					<div
						@dragover="handleDragover"
						@dragleave="handleDragleave"
						@drop="handleDrop"
						:class="[
							'p-12 md:p-16 border-2 border-dashed transition-all cursor-pointer',
							isDragging
								? 'border-primary/70 bg-primary/5 shadow-[var(--glow)]'
								: 'border-border hover:border-primary/40 hover:bg-card/60'
						]"
					>
						<div class="flex flex-col items-center justify-center text-center space-y-4">
							<div :class="['w-16 h-16 rounded-2xl flex items-center justify-center transition-all', isDragging ? 'bg-primary/20 scale-110' : 'bg-primary/10']">
								<Upload :class="['w-8 h-8 text-primary', isDragging ? 'scale-125' : '']" />
							</div>
							<div>
								<h3 class="text-2xl font-bold text-foreground mb-1">Drop your file here</h3>
								<p class="text-foreground/60">or click to browse your computer</p>
							</div>
							<button
								@click="triggerFileInput"
								class="px-6 py-3 rounded-xl bg-gradient-to-r from-primary to-secondary hover:shadow-[var(--glow)] text-foreground font-semibold transition-all duration-200 hover:scale-105"
							>
								Select File
							</button>
							<p class="text-xs text-foreground/50 pt-2">Supported formats: CSV, JSON • Max size: 500MB</p>
						</div>
						<input
							ref="fileInputRef"
							type="file"
							accept=".csv,.json"
							@change="handleFileSelect"
							class="hidden"
						/>
					</div>

					<!-- File Preview -->
					<transition name="slide-fade">
						<div v-if="selectedFile" class="p-6 border-t border-border bg-card/70 backdrop-blur-md">
							<div class="flex items-center justify-between">
								<div class="flex items-center gap-4">
									<div class="w-12 h-12 rounded-lg bg-primary/15 flex items-center justify-center">
										<FileText class="w-6 h-6 text-primary" />
									</div>
									<div>
										<p class="font-semibold text-foreground">{{ fileName }}</p>
										<p class="text-sm text-foreground/60">{{ fileSize }}</p>
									</div>
								</div>
								<button
									@click="resetUpload"
									class="text-foreground/60 hover:text-foreground transition"
									type="button"
								>
									<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
									</svg>
								</button>
							</div>
						</div>
					</transition>

					<!-- Action Buttons -->
					<transition name="slide-fade">
						<div v-if="selectedFile" class="p-6 border-t border-border bg-card/50 backdrop-blur-md flex gap-3">
							<button
								@click="handleUpload"
								:disabled="isLoading"
								class="flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-primary to-secondary hover:shadow-[var(--glow)] text-foreground font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105"
							>
								<Upload class="w-5 h-5" />
								<span>Upload File</span>
							</button>
							<button
								@click="resetUpload"
								class="px-6 py-3 rounded-xl border border-border hover:border-primary/40 text-foreground font-semibold transition"
								type="button"
							>
								Cancel
							</button>
						</div>
					</transition>
				</div>

				<!-- Info Cards -->
				<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
					<div class="rounded-xl border border-border/30 bg-card/30 backdrop-blur-sm p-4">
						<h4 class="font-semibold text-foreground text-sm mb-2">What happens next?</h4>
						<p class="text-xs text-foreground/60">Your file will be analyzed for data quality issues and cleaned automatically.</p>
					</div>
					<div class="rounded-xl border border-border/30 bg-card/30 backdrop-blur-sm p-4">
						<h4 class="font-semibold text-foreground text-sm mb-2">Supported Formats</h4>
						<p class="text-xs text-foreground/60">CSV and JSON files. We'll handle the rest.</p>
					</div>
					<div class="rounded-xl border border-border/30 bg-card/30 backdrop-blur-sm p-4">
						<h4 class="font-semibold text-foreground text-sm mb-2">Processing Time</h4>
						<p class="text-xs text-foreground/60">Typically 2-5 minutes depending on file size.</p>
					</div>
				</div>
			</div>

			<!-- Uploading State -->
			<div v-else-if="uploadStatus === 'uploading'" class="rounded-2xl border border-border bg-card/40 backdrop-blur-sm p-12 md:p-16 text-center">
				<div class="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/15 mb-6">
					<Loader2 class="w-8 h-8 text-primary animate-spin" />
				</div>
				<h3 class="text-2xl font-bold text-foreground mb-2">Uploading your file...</h3>
				<p class="text-foreground/60 mb-8 max-w-lg mx-auto">
					Please don't close this window. We're securely uploading your data.
				</p>
				<div class="max-w-sm mx-auto">
					<div class="flex items-end gap-3 mb-3">
						<div class="flex-1 bg-border/30 rounded-full h-3 overflow-hidden">
							<div
								class="bg-gradient-to-r from-primary to-secondary h-full transition-all duration-300"
								:style="{ width: uploadProgress + '%' }"
							/>
						</div>
						<span class="text-sm font-semibold text-foreground/80 min-w-max">{{ uploadProgress }}%</span>
					</div>
					<p class="text-xs text-foreground/50">{{ fileName }}</p>
				</div>
			</div>

			<!-- Success State -->
			<div v-else-if="uploadStatus === 'success'" class="rounded-2xl border border-success/30 bg-success/5 backdrop-blur-sm p-12 md:p-16 text-center">
				<div class="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-success/15 mb-6">
					<CheckCircle class="w-8 h-8 text-success" />
				</div>
				<h3 class="text-2xl font-bold text-foreground mb-2">Upload Successful! 🎉</h3>
				<p class="text-foreground/60 mb-8 max-w-lg mx-auto">
					Your file has been uploaded successfully. The processing pipeline is starting now.
				</p>
				<div class="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-success/20 text-success text-sm font-semibold mb-8">
					<svg class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
					</svg>
					Processing...
				</div>
				<p class="text-xs text-foreground/60 mb-6">Redirecting to pipelines in a moment...</p>
			</div>

			<!-- Error State -->
			<div v-else-if="uploadStatus === 'error'" class="rounded-2xl border border-destructive/30 bg-destructive/5 backdrop-blur-sm p-12 md:p-16 text-center">
				<div class="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-destructive/15 mb-6">
					<AlertCircle class="w-8 h-8 text-destructive" />
				</div>
				<h3 class="text-2xl font-bold text-foreground mb-2">Upload Failed</h3>
				<p class="text-destructive mb-8 max-w-lg mx-auto">
					{{ errorMessage }}
				</p>
				<div class="space-y-3">
					<button
						@click="resetUpload"
						class="w-full md:w-auto px-6 py-3 rounded-xl bg-gradient-to-r from-primary to-secondary hover:shadow-[var(--glow)] text-foreground font-semibold transition-all hover:scale-105"
					>
						Try Again
					</button>
					<p class="text-xs text-foreground/60 pt-4">
						Having trouble? <a href="#" class="text-primary hover:text-primary/80 underline">Contact support</a>
					</p>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped></style>
