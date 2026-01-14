// otp-login-form.vue
<script setup lang="ts">
import { useMutation } from '@tanstack/vue-query';
// Import useMutation
import { toTypedSchema } from '@vee-validate/zod';
import { ArrowRight } from 'lucide-vue-next';
import { useForm } from 'vee-validate';
import { nextTick, ref, watch } from 'vue';
import { onUnmounted } from 'vue';
import { toast } from 'vue-sonner';
import * as z from 'zod';



import OtpInput from '@/core/components/otp-input/OTPInput.vue';
import { Button } from '@/core/components/ui/button';
import { FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/core/components/ui/form';
import { apiClient } from '@/core/configs/axios-instance.config';



// Run immediately on component mount if phoneNumber is already set

// Clear interval on component unmount
import { AuthServiceError, authService } from '@/modules/auth/services/auth.service';





const props = defineProps<{
  phoneNumber: string; // Phone number passed from parent
}>();

const emit = defineEmits<{
  (e: 'authenticated'): void;
  (e: 'resend-requested', phoneNumber: string): void;
  (e: 'back-to-phone'): void; // New event to go back to phone input
}>();

// Configuration
const OTP_LENGTH = 4;
const RESEND_COOLDOWN_SECONDS = 60; // Cooldown for resending OTP

// Refs
const otpInputRef = ref<InstanceType<typeof OtpInput> | null>(null);
const resendCooldown = ref(0);
let resendInterval: number | undefined;

// Zod schema with custom validation
const otpSchema = toTypedSchema(
  z.object({
    otp: z
      .string()
      .min(1, 'OTP код қажет')
      .length(OTP_LENGTH, `${OTP_LENGTH} сан енгізіңіз`)
      .regex(/^\d+$/, 'Тек сандар енгізуге болады'),
  })
);

// Vee-validate form
const { handleSubmit, setFieldError, values, resetForm, setFieldValue, meta } = useForm({
  validationSchema: otpSchema,
  initialValues: { otp: '' },
});

// useMutation for verifying OTP
const verifyOtpMutation = useMutation({
  mutationFn: ({ phone, otp }: { phone: string; otp: string }) => authService.verifyOtp(phone, otp),
  onSuccess: (data: any) => {
    // Persist tokens so closing/reopening tab keeps the session
    try {
      if (data && (data.accessToken || data.refreshToken)) {
        localStorage.setItem('authTokens', JSON.stringify(data));
        if (data.accessToken) {
          apiClient.defaults.headers.common['Authorization'] = `Bearer ${data.accessToken}`;
        }
      }
    } catch (e) {
      console.warn('Failed to persist auth tokens', e);
    }

    toast.success('Сәтті!', {
      description: 'OTP код дұрыс енгізілді. Кіру жүзеге асырылды.',
    });
    emit('authenticated');
  },
  onError: (error: AuthServiceError) => {
    console.error('Error verifying OTP:', error);
    setFieldError('otp', error.message || 'OTP дұрыс емес. Қайта көріңіз.');
    // Clear the OTP input and refocus on error
    nextTick(() => {
      otpInputRef.value?.clear();
    });
    toast.error('Қате', {
      description: error.message || 'OTP код тексеру кезінде қате орын алды.',
    });
  },
});

// useMutation for resending OTP
const resendOtpMutation = useMutation({
  mutationFn: (phone: string) => authService.sendOtp(phone),
  onSuccess: () => {
    toast.success('Код жіберілді', {
      description: 'Жаңа OTP код жіберілді.',
    });
    emit('resend-requested', props.phoneNumber); // Inform parent about resend
    startResendCooldown(); // Restart cooldown
    resetForm(); // Clear the form on resend
    nextTick(() => {
      otpInputRef.value?.clear();
      otpInputRef.value?.focus();
    });
  },
  onError: (error: AuthServiceError) => {
    console.error('Error resending OTP:', error);
    toast.error('Қате', {
      description: error.message || 'Код жіберу кезінде қате орын алды.',
    });
  },
});

// Submit handler
const onSubmit = handleSubmit(async (vals) => {
  if (verifyOtpMutation.isPending.value) return; // Prevent double submission

  // Ensure phone number is available before attempting to verify
  if (!props.phoneNumber) {
    toast.error('Қате', {
      description: 'Телефон нөмірі табылмады. Қайта кіруге тырысыңыз.',
    });
    return;
  }
  verifyOtpMutation.mutate({ phone: props.phoneNumber, otp: vals.otp });
});

// Auto-submit when OTP is complete
const handleOtpComplete = (value: string) => {
  setFieldValue('otp', value); // Ensure the field value is updated
  if (!verifyOtpMutation.isPending.value) {
    onSubmit();
  }
};

// Handle OTP change
const handleOtpChange = (value: string) => {
  setFieldValue('otp', value);
  // Clear any previous errors when user starts typing
  if (value.length > 0 && !meta.value.valid) {
    setFieldError('otp', undefined);
  }
};

// Resend OTP functionality
const handleResend = async () => {
  if (resendOtpMutation.isPending.value || resendCooldown.value > 0) return;

  if (!props.phoneNumber) {
    toast.error('Қате', {
      description: 'Телефон нөмірі жоқ. Артқа қайтып, телефон нөмірін енгізіңіз.',
    });
    return;
  }
  resendOtpMutation.mutate(props.phoneNumber);
};

// Cooldown logic for resend button
const startResendCooldown = () => {
  resendCooldown.value = RESEND_COOLDOWN_SECONDS;
  clearInterval(resendInterval); // Clear any existing interval
  resendInterval = setInterval(() => {
    resendCooldown.value--;
    if (resendCooldown.value <= 0) {
      clearInterval(resendInterval);
      resendInterval = undefined;
    }
  }, 1000) as unknown as number; // Type assertion for setInterval return
};

// Start cooldown immediately when component is mounted or phone number is available
watch(() => props.phoneNumber, (newPhone) => {
  if (newPhone) {
    startResendCooldown();
  }
}, { immediate: true }); // Run immediately on component mount if phoneNumber is already set



onUnmounted(() => {
  if (resendInterval) {
    clearInterval(resendInterval);
  }
});
</script>

<template>
	<form
		@submit.prevent="onSubmit"
		class="space-y-6"
	>
		<FormField
			name="otp"
			v-slot="{ componentField, meta }"
		>
			<FormItem>
				<FormLabel class="sr-only"> OTP коды ({{ OTP_LENGTH }} сан) </FormLabel>

				<FormControl>
					<OtpInput
						ref="otpInputRef"
						:model-value="componentField.modelValue"
						:length="OTP_LENGTH"
						:disabled="verifyOtpMutation.isPending.value"
						:aria-invalid="meta.valid === false && meta.touched"
						container-class="justify-center"
						input-class="size-16 text-xl"
						@update:model-value="componentField['onUpdate:modelValue']"
						@input="componentField.onInput"
						@complete="handleOtpComplete"
						@change="handleOtpChange"
					/>
				</FormControl>
				<FormMessage class="text-center" />
			</FormItem>
		</FormField>

		<div class="flex flex-col items-center space-y-3">
			<Button
				type="button"
				variant="ghost"
				size="sm"
				@click="handleResend"
				:disabled="resendOtpMutation.isPending.value || resendCooldown > 0"
			>
				<span v-if="resendOtpMutation.isPending.value">Жіберілуде...</span>
				<span v-else-if="resendCooldown > 0"> Қайта жіберу ({{ resendCooldown }}с) </span>
				<span v-else>Қайта жіберу</span>
			</Button>

			<button
				type="submit"
				:disabled="verifyOtpMutation.isPending.value || values.otp?.length !== OTP_LENGTH"
				:class="[
        'w-full flex items-center justify-center gap-3 py-4 px-6 rounded-2xl font-semibold text-white transition-all duration-200 text-lg shadow-lg',
        values.otp?.length === OTP_LENGTH && !verifyOtpMutation.isPending
          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]'
          : 'bg-gray-300 cursor-not-allowed'
      ]"
			>
				<Loader2
					v-if="verifyOtpMutation.isPending.value"
					class="w-5 h-5 animate-spin"
				/>
				<template v-else>
					<span>Кіру</span>
					<ArrowRight class="w-5 h-5 transition-transform group-hover:translate-x-1" />
				</template>
			</button>
		</div>
	</form>
</template>
