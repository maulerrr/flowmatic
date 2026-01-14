<script setup lang="ts">
import { useMutation } from '@tanstack/vue-query';
import { ArrowRight, Loader2, Phone } from 'lucide-vue-next';
import { computed, ref } from 'vue';
import { toast } from 'vue-sonner';



import { AuthServiceError, authService } from '@/modules/auth/services/auth.service';





const emit = defineEmits<{
  sent: [phone: string]
}>();

const phoneNumber = ref('');

const isValidPhone = computed(() => {
  const cleanPhone = phoneNumber.value.replace(/\D/g, '');
  return cleanPhone.length >= 10;
});

const formattedPhone = computed({
  get: () => phoneNumber.value,
  set: (value: string) => {
    // Remove all non-digits
    const digits = value.replace(/\D/g, '');

    // Format as +7 (XXX) XXX-XX-XX
    if (digits.length === 0) {
      phoneNumber.value = '';
    } else {
      let formatted = '+7';
      if (digits.length > 1) {
        formatted += ` (${digits.slice(1, 4)}`;
        if (digits.length > 4) {
          formatted += `) ${digits.slice(4, 7)}`;
          if (digits.length > 7) {
            formatted += `-${digits.slice(7, 9)}`;
            if (digits.length > 9) {
              formatted += `-${digits.slice(9, 11)}`;
            }
          }
        }
      }
      phoneNumber.value = formatted;
    }
  }
});

const {mutate, isPending: isLoading, error} = useMutation({
  mutationFn: (phone: string) => authService.sendOtp(phone),
  onSuccess: (_: unknown, variables: string) => {
    toast.success('Код жіберілді', {
      description: `OTP коды ${variables} нөміріне жіберілді.`,
    });
    emit('sent', variables); // Emit the phone number on success
  },
  onError: (error: AuthServiceError) => {
    console.error('Error sending phone number:', error);
    toast.error('Қате', {
      description: error.message || 'Телефон нөмірін жіберу кезінде қате орын алды.',
    });
  },
});

async function handleSubmit() {
  if (!isValidPhone.value) {
    return;
  }

  const cleanPhone = phoneNumber.value.replace(/\D/g, '');
  const number = '+7' + cleanPhone.slice(1)

  mutate(number, {onSuccess: () => {    emit('sent', number);
}})

}
</script>

<template>
	<form
		@submit.prevent="handleSubmit"
		class="space-y-6"
	>
		<!-- Phone Input -->
		<div class="space-y-2">
			<label
				for="phone"
				class="block font-semibold text-gray-700 text-sm"
			>
				Телефон нөмірі
			</label>
			<div class="relative">
				<div class="top-1/2 left-4 absolute flex items-center gap-2 -translate-y-1/2 transform">
					<div
						class="flex justify-center items-center bg-gradient-to-br from-green-500 to-emerald-600 rounded-md w-6 h-6"
					>
						<Phone class="w-3 h-3 text-white" />
					</div>
				</div>
				<input
					id="phone"
					v-model="formattedPhone"
					type="tel"
					placeholder="+7 (___) ___-__-__"
					:class="[
            'w-full pl-14 pr-4 py-4 bg-white border rounded-2xl text-gray-900 placeholder-gray-400 transition-all duration-200 focus:outline-none text-lg font-medium',
            error ? 'border-red-300 focus:border-red-500 focus:ring-4 focus:ring-red-500/10' :
            'border-slate-200 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10',
            'hover:border-gray-300'
          ]"
					:disabled="isLoading"
				/>
			</div>

			<Transition name="fade">
				<p
					v-if="error"
					class="flex items-center gap-2 text-red-600 text-sm"
				>
					<span class="flex justify-center items-center bg-red-100 rounded-full w-4 h-4">
						<span class="bg-red-500 rounded-full w-2 h-2"></span>
					</span>
					{{ error }}
				</p>
			</Transition>
		</div>

		<!-- Submit Button -->
		<button
			type="submit"
			:disabled="!isValidPhone || isLoading"
			:class="[
        'w-full flex items-center justify-center gap-3 py-4 px-6 rounded-2xl font-semibold text-white transition-all duration-200 text-lg shadow-lg',
        isValidPhone && !isLoading
          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]'
          : 'bg-gray-300 cursor-not-allowed'
      ]"
		>
			<Loader2
				v-if="isLoading"
				class="w-5 h-5 animate-spin"
			/>
			<template v-else>
				<span>SMS код жіберу</span>
				<ArrowRight class="w-5 h-5 transition-transform group-hover:translate-x-1" />
			</template>
		</button>

		<!-- Help Text -->
		<div class="space-y-2 text-center">
			<p class="text-gray-500 text-sm">SMS код 2-3 минут ішінде жіберіледі</p>
			<p class="text-gray-400 text-xs">Қайталанбайтын код сіздің телефоныңызға жіберіледі</p>
		</div>
	</form>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: all 0.2s ease-out;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
