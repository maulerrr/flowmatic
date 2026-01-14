<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';



import { cn } from '@/core/utils/tailwind.utils';





interface Props {
  modelValue?: string
  length?: number
  disabled?: boolean
  placeholder?: string
  autoFocus?: boolean
  pushOnComplete?: boolean
  containerClass?: string
  inputClass?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: '',
  length: 4,
  disabled: false,
  placeholder: '',
  autoFocus: true,
  pushOnComplete: false,
  containerClass: '',
  inputClass: ''
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  'complete': [value: string]
  'change': [value: string]
  'focus': [index: number]
  'blur': [index: number]
  'input': [value: string]
}>()

// Reactive refs
const digits = ref<string[]>(Array(props.length).fill(''))
const inputs = ref<(HTMLInputElement | null)[]>(Array(props.length).fill(null))
const currentFocusIndex = ref<number>(-1)

// Computed
const isComplete = computed(() =>
  digits.value.every(digit => digit !== '') && digits.value.join('').length === props.length
)

// Watch modelValue changes and sync with digits
watch(
  () => props.modelValue,
  (newValue) => {
    const chars = (newValue || '').split('').slice(0, props.length)
    const newDigits = Array(props.length).fill('')

    chars.forEach((char, index) => {
      if (/^\d$/.test(char)) {
        newDigits[index] = char
      }
    })

    digits.value = newDigits

    // Update DOM inputs to match
    nextTick(() => {
      inputs.value.forEach((input, index) => {
        if (input) {
          input.value = newDigits[index] || ''
        }
      })
    })
  },
  { immediate: true }
)

// Watch digits changes and emit updates
watch(
  digits,
  (newDigits) => {
    const value = newDigits.join('')
    emit('update:modelValue', value)
    emit('change', value)
    emit('input', value) // For vee-validate compatibility

    if (isComplete.value) {
      emit('complete', value)
      if (props.pushOnComplete) {
        // Blur all inputs when complete
        inputs.value.forEach(input => input?.blur())
      }
    }
  },
  { deep: true }
)

// Input handlers
const handleInput = async (event: Event, index: number) => {
  if (props.disabled) return

  const target = event.target as HTMLInputElement
  const value = target.value.replace(/\D/g, '') // Only digits

  if (!value) {
    digits.value[index] = ''
    return
  }

  // Handle single digit input
  if (value.length === 1) {
    digits.value[index] = value
    target.value = value

    // Move to next input
    await nextTick()
    const nextIndex = findNextEmptyIndex(index)
    if (nextIndex !== -1) {
      focusInput(nextIndex)
    }
    return
  }

  // Handle pasted content or multiple digits
  const chars = value.split('')
  let startIndex = index

  // Clear current input first
  digits.value[index] = ''

  // Fill from current position
  for (let i = 0; i < chars.length && startIndex < props.length; i++) {
    if (/^\d$/.test(chars[i])) {
      digits.value[startIndex] = chars[i]
      startIndex++
    }
  }

  await nextTick()

  // Update DOM inputs
  inputs.value.forEach((input, idx) => {
    if (input) {
      input.value = digits.value[idx] || ''
    }
  })

  // Focus next empty or last filled
  const nextEmpty = findNextEmptyIndex(index)
  if (nextEmpty !== -1) {
    focusInput(nextEmpty)
  } else if (isComplete.value) {
    focusInput(props.length - 1)
  }
}

const handleKeyDown = async (event: KeyboardEvent, index: number) => {
  if (props.disabled) return

  const target = event.target as HTMLInputElement

  switch (event.key) {
    case 'Backspace':
      event.preventDefault()
      if (target.value) {
        // Clear current input
        digits.value[index] = ''
        target.value = ''
      } else {
        // Move to previous input and clear it
        const prevIndex = findPreviousFilledIndex(index)
        if (prevIndex !== -1) {
          digits.value[prevIndex] = ''
          focusInput(prevIndex)
        }
      }
      break

    case 'Delete':
      event.preventDefault()
      digits.value[index] = ''
      target.value = ''
      break

    case 'ArrowLeft':
      event.preventDefault()
      const prevIndex = Math.max(0, index - 1)
      focusInput(prevIndex)
      break

    case 'ArrowRight':
      event.preventDefault()
      const nextIndex = Math.min(props.length - 1, index + 1)
      focusInput(nextIndex)
      break

    case 'Home':
      event.preventDefault()
      focusInput(0)
      break

    case 'End':
      event.preventDefault()
      focusInput(props.length - 1)
      break

    case 'Enter':
      if (isComplete.value) {
        emit('complete', digits.value.join(''))
      }
      break

    // Prevent non-numeric input
    default:
      if (!/^\d$/.test(event.key) && !['Tab', 'Shift'].includes(event.key)) {
        event.preventDefault()
      }
      break
  }
}

const handleFocus = (event: FocusEvent, index: number) => {
  currentFocusIndex.value = index
  const target = event.target as HTMLInputElement

  // Select all content on focus for easy replacement
  nextTick(() => {
    target.select()
  })

  emit('focus', index)
}

const handleBlur = (event: FocusEvent, index: number) => {
  currentFocusIndex.value = -1
  emit('blur', index)
}

const handlePaste = async (event: ClipboardEvent, index: number) => {
  event.preventDefault()
  if (props.disabled) return

  const pasteData = event.clipboardData?.getData('text') || ''
  const digits_only = pasteData.replace(/\D/g, '')

  if (!digits_only) return

  const chars = digits_only.split('').slice(0, props.length - index)

  // Fill from current position
  chars.forEach((char, offset) => {
    const targetIndex = index + offset
    if (targetIndex < props.length) {
      digits.value[targetIndex] = char
    }
  })

  await nextTick()

  // Update DOM
  inputs.value.forEach((input, idx) => {
    if (input) {
      input.value = digits.value[idx] || ''
    }
  })

  // Focus appropriate input
  const newFocusIndex = Math.min(index + chars.length, props.length - 1)
  focusInput(newFocusIndex)
}

// Helper functions
const focusInput = (index: number) => {
  if (index >= 0 && index < props.length && inputs.value[index]) {
    inputs.value[index]?.focus()
  }
}

const findNextEmptyIndex = (currentIndex: number): number => {
  for (let i = currentIndex + 1; i < props.length; i++) {
    if (!digits.value[i]) return i
  }
  return -1
}

const findPreviousFilledIndex = (currentIndex: number): number => {
  for (let i = currentIndex - 1; i >= 0; i--) {
    if (digits.value[i]) return i
  }
  return Math.max(0, currentIndex - 1)
}

// Public methods
const clear = () => {
  digits.value = Array(props.length).fill('')
  inputs.value.forEach(input => {
    if (input) input.value = ''
  })
  if (props.autoFocus) {
    focusInput(0)
  }
}

const focus = (index: number = 0) => {
  focusInput(index)
}

// Expose public methods
defineExpose({
  clear,
  focus,
  digits: digits.value,
  isComplete
})

// Auto-focus first input on mount
onMounted(() => {
  if (props.autoFocus && !props.disabled) {
    nextTick(() => {
      focusInput(0)
    })
  }
})
</script>

<template>
	<div
		:class="cn(
      'flex items-center gap-2',
      containerClass
    )"
		role="group"
		:aria-label="`Enter ${length} digit code`"
	>
		<template
			v-for="(_, index) in length"
			:key="index"
		>
			<input
				:ref="(el) => inputs[index] = el as HTMLInputElement"
				:class="cn(
          // Base styles
          'relative flex h-12 w-12 items-center justify-center rounded-md border border-input bg-background text-center text-lg font-medium text-foreground transition-all',
          // Focus styles
          'focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20',
          // Hover styles
          'hover:border-primary/50',
          // Disabled styles
          'disabled:cursor-not-allowed disabled:opacity-50',
          // Error styles (can be extended)
          'aria-[invalid=true]:border-destructive aria-[invalid=true]:focus:ring-destructive/20',
          // Filled state
          digits[index] && '',
          inputClass
        )"
				type="text"
				inputmode="numeric"
				pattern="[0-9]*"
				maxlength="1"
				:value="digits[index]"
				:disabled="disabled"
				:placeholder="placeholder"
				:aria-label="`Digit ${index + 1} of ${length}`"
				:aria-describedby="`otp-input-${index}`"
				autocomplete="one-time-code"
				@input="(e) => handleInput(e, index)"
				@keydown="(e) => handleKeyDown(e, index)"
				@focus="(e) => handleFocus(e, index)"
				@blur="(e) => handleBlur(e, index)"
				@paste="(e) => handlePaste(e, index)"
			/>
		</template>
	</div>
</template>
