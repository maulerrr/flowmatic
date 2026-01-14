<!-- FileUpload.vue -->
<template>
	<div class="space-y-3">
		<!-- Label + clear button -->
		<div class="flex items-center justify-between">
			<Label
				:for="inputId"
				class="text-sm font-medium text-gray-700 dark:text-gray-300"
			>
				{{ title }}
				<span
					v-if="required"
					class="text-red-500 ml-1"
					>*</span
				>
			</Label>
			<Button
				v-if="modelValue && showClearButton"
				variant="ghost"
				size="sm"
				@click="clearFile"
				class="text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 p-1 h-auto"
			>
				<X class="w-4 h-4" />
			</Button>
		</div>

		<!-- Optional description -->
		<p
			v-if="description"
			class="text-sm text-gray-500 dark:text-gray-400"
		>
			{{ description }}
		</p>

		<!-- Drop zone -->
		<div
			ref="dropZone"
			role="button"
			tabindex="0"
			:aria-disabled="disabled"
			:aria-invalid="hasError"
			:class="[
        'relative border-2 border-dashed rounded-lg transition-all duration-300 cursor-pointer',
        isDragging
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 drag-over'
          : 'border-gray-300 dark:border-gray-600',
        'hover:border-blue-400 dark:hover:border-blue-500',
        'focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20',
        disabled ? 'opacity-50 cursor-not-allowed' : '',
        hasError ? 'border-red-400 bg-red-50 dark:bg-red-900/20' : ''
      ]"
			@click="triggerFileInput"
			@keydown.enter.prevent="triggerFileInput"
			@dragover.prevent="handleDragOver"
			@dragleave.prevent="handleDragLeave"
			@drop.prevent="handleDrop"
		>
			<!-- Invisible file input (stops propagation) -->
			<input
				:id="inputId"
				ref="fileInput"
				type="file"
				:accept="accept"
				:multiple="multiple"
				:disabled="disabled"
				class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
				@click.stop
				@change="handleFileSelect"
			/>

			<!-- Main upload UI -->
			<div class="flex flex-col items-center justify-center p-6 text-center min-h-[120px]">
				<div class="mb-4">
					<component
						:is="getIcon()"
						:class="[
              'w-12 h-12 transition-colors',
              isDragging || hasFile
                ? 'text-blue-500'
                : 'text-gray-400 dark:text-gray-500'
            ]"
					/>
				</div>
				<div class="space-y-2">
					<p class="text-base font-medium text-gray-900 dark:text-gray-100">
						{{ getMainText() }}
					</p>
					<p class="text-sm text-gray-500 dark:text-gray-400">
						{{ getSubText() }}
					</p>
				</div>

				<!-- Single file info -->
				<div
					v-if="fileInfo && !multiple"
					class="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-md w-full max-w-sm file-preview"
				>
					<div class="flex items-center space-x-3">
						<FileIcon class="w-5 h-5 text-blue-500 flex-shrink-0" />
						<div class="flex-1 min-w-0">
							<p class="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
								{{ fileInfo.name }}
							</p>
							<p class="text-xs text-gray-500 dark:text-gray-400">
								{{ formatFileSize(fileInfo.size) }}
							</p>
						</div>
					</div>
				</div>

				<!-- Multiple files list -->
				<div
					v-if="filesList.length && multiple"
					class="mt-4 w-full max-w-sm space-y-2 file-preview"
				>
					<div
						v-for="(file, idx) in filesList"
						:key="idx"
						class="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded-md"
					>
						<div class="flex items-center space-x-2 flex-1 min-w-0">
							<FileIcon class="w-4 h-4 text-blue-500 flex-shrink-0" />
							<div class="flex-1 min-w-0">
								<p class="text-sm text-gray-900 dark:text-gray-100 truncate">
									{{ file.name }}
								</p>
								<p class="text-xs text-gray-500 dark:text-gray-400">
									{{ formatFileSize(file.size) }}
								</p>
							</div>
						</div>
						<Button
							variant="ghost"
							size="sm"
							@click.stop="removeFile(idx)"
							class="text-red-600 hover:text-red-700 hover:bg-red-100 dark:hover:bg-red-900/20 p-1 h-auto"
						>
							<X class="w-3 h-3" />
						</Button>
					</div>
				</div>
			</div>

			<!-- Loading overlay -->
			<div
				v-if="loading"
				class="absolute inset-0 bg-white/80 dark:bg-gray-800/80 flex items-center justify-center rounded-lg"
			>
				<div class="flex items-center space-x-2">
					<div
						class="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"
					></div>
					<span class="text-sm text-gray-600 dark:text-gray-400">{{ loadingText }}</span>
				</div>
			</div>
		</div>

		<!-- Error message -->
		<p
			v-if="errorMessage"
			role="alert"
			class="text-sm text-red-600 dark:text-red-400 flex items-center space-x-1"
		>
			<AlertCircle class="w-4 h-4" />
			<span>{{ errorMessage }}</span>
		</p>

		<!-- File constraints -->
		<div
			v-if="showConstraints"
			class="text-xs text-gray-500 dark:text-gray-400 space-y-1"
		>
			<p v-if="maxSize">Максимальный размер: {{ formatFileSize(maxSize) }}</p>
			<p v-if="accept">Допустимые форматы: {{ formattedAccept }}</p>
			<p v-if="multiple && maxFiles">Максимум файлов: {{ maxFiles }}</p>
		</div>
	</div>
</template>

<script setup lang="ts">
import { AlertCircle, CheckCircle, FileAudio, File as FileIcon, FileImage, FileVideo, Upload, X } from 'lucide-vue-next';
import { computed, ref, watch } from 'vue';



import { Button } from '@/core/components/ui/button';
import { Label } from '@/core/components/ui/label';





interface FileUploadProps {
  modelValue?: File | File[] | string | null
  title: string
  description?: string
  accept?: string
  multiple?: boolean
  maxSize?: number
  maxFiles?: number
  required?: boolean
  disabled?: boolean
  loading?: boolean
  loadingText?: string
  showClearButton?: boolean
  showConstraints?: boolean
  customValidator?: (file: File) => string | null
}

const props = withDefaults(defineProps<FileUploadProps>(), {
  accept: '*/*',
  multiple: false,
  maxSize: 10 * 1024 * 1024,
  maxFiles: 5,
  required: false,
  disabled: false,
  loading: false,
  loadingText: 'Загрузка...',
  showClearButton: true,
  showConstraints: true
})

const emit = defineEmits<{
  'update:modelValue': [File | File[] | null]
  'filesSelected': [File[]]
  'error': [string]
  'cleared': []
}>()

// stable ID
const inputId = `file-upload-${Math.random().toString(36).substring(2, 9)}`

// refs & state
const fileInput = ref<HTMLInputElement|null>(null)
const dropZone = ref<HTMLElement|null>(null)
const isDragging = ref(false)
const errorMessage = ref('')

// computed
const hasFile = computed(() =>
  props.multiple
    ? Array.isArray(props.modelValue) && props.modelValue.length > 0
    : !!props.modelValue
)

const hasError = computed(() => Boolean(errorMessage.value))

const fileInfo = computed(() => {
  if (props.multiple || !(props.modelValue instanceof File)) return null
  return {
    name: props.modelValue.name,
    size: props.modelValue.size,
    type: props.modelValue.type
  }
})

const filesList = computed<File[]>(() =>
  props.multiple && Array.isArray(props.modelValue)
    ? (props.modelValue.filter(f => f instanceof File) as File[])
    : []
)

const formattedAccept = computed(() =>
  props.accept
    .split(',')
    .map(s => s.trim())
    .join(', ')
)

const getIcon = () => {
  if (hasFile.value) return CheckCircle
  if (props.accept.includes('image/')) return FileImage
  if (props.accept.includes('audio/')) return FileAudio
  if (props.accept.includes('video/')) return FileVideo
  return Upload
}

const getMainText = () => {
  if (hasFile.value) {
    return props.multiple
      ? `Выбрано файлов: ${filesList.value.length}`
      : 'Файл выбран'
  }
  return isDragging.value
    ? 'Отпустите для загрузки'
    : 'Перетащите файл или нажмите для выбора'
}

const getSubText = () => {
  if (hasFile.value) return 'Нажмите для изменения'
  const sizeTxt = props.maxSize ? ` до ${formatFileSize(props.maxSize)}` : ''
  const multiTxt = props.multiple ? ' (можно выбрать несколько)' : ''
  return `Файл${sizeTxt}${multiTxt}`
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes','KB','MB','GB']
  const i = Math.floor(Math.log(bytes)/Math.log(k))
  return (bytes/Math.pow(k,i)).toFixed(2) + ' ' + sizes[i]
}

const validateFile = (file: File): string|null => {
  if (props.maxSize && file.size > props.maxSize) {
    return `Файл слишком большой. Максимальный размер: ${formatFileSize(props.maxSize)}`
  }
  if (props.accept !== '*/*') {
    const types = props.accept.split(',').map(t => t.trim())
    const ok = types.some(t =>
      t.startsWith('.')
        ? file.name.toLowerCase().endsWith(t.toLowerCase())
        : new RegExp('^'+t.replace('*','.*')+'$').test(file.type)
    )
    if (!ok) return `Неподдерживаемый тип файла. Допустимые: ${formattedAccept.value}`
  }
  return props.customValidator?.(file) || null
}

const validateFiles = (files: File[]) => {
  if (props.multiple && props.maxFiles && files.length > props.maxFiles) {
    return `Слишком много файлов. Максимум: ${props.maxFiles}`
  }
  for (const f of files) {
    const err = validateFile(f)
    if (err) return err
  }
  return null
}

const handleFiles = (files: FileList|File[]) => {
  const arr = Array.from(files)
  const err = validateFiles(arr)
  if (err) {
    errorMessage.value = err
    emit('error', err)
    return
  }
  errorMessage.value = ''
  if (props.multiple) {
    emit('update:modelValue', arr)
  } else {
    emit('update:modelValue', arr[0] || null)
  }
  emit('filesSelected', arr)
}

const handleFileSelect = (e: Event) => {
  const inEl = e.target as HTMLInputElement
  if (inEl.files?.length) handleFiles(inEl.files)
}

const handleDragOver = () => { if (!props.disabled) isDragging.value = true }
const handleDragLeave = (e: DragEvent) => {
  if (props.disabled) return
  const rect = dropZone.value?.getBoundingClientRect()
  if (rect && (e.clientX<rect.left || e.clientX>rect.right || e.clientY<rect.top || e.clientY>rect.bottom)) {
    isDragging.value = false
  }
}
const handleDrop = (e: DragEvent) => {
  if (props.disabled) return
  isDragging.value = false
  if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files)
}

const triggerFileInput = () => {
  if (!props.disabled && !props.loading) fileInput.value?.click()
}

const clearFile = () => {
  errorMessage.value = ''
  emit('update:modelValue', props.multiple ? [] : null)
  emit('cleared')
  if (fileInput.value) fileInput.value.value = ''
}

const removeFile = (idx: number) => {
  if (!props.multiple || !Array.isArray(props.modelValue)) return
  const copy = [...props.modelValue]
  copy.splice(idx, 1)
  emit('update:modelValue', copy)
}

watch(() => props.modelValue, val => {
  if (!val || (Array.isArray(val) && val.length === 0)) errorMessage.value = ''
})
</script>

<style scoped>
/* fade-in for previews */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px) }
  to   { opacity: 1; transform: translateY(0) }
}
.file-preview { animation: fadeIn 0.3s ease-out }
.drag-over { transform: scale(1.02) }
</style>
