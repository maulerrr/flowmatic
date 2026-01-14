<script setup lang="ts">
import { ArrowLeft, ArrowRight, CheckCircle, Shield, Smartphone } from 'lucide-vue-next';
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';



import OtpLoginForm from '@/modules/auth/components/otp-login-form.vue';
import PhoneLoginForm from '@/modules/auth/components/phone-login-form.vue';





const router = useRouter();
const stage = ref<'phone' | 'otp'>('phone');
const phoneNumber = ref<string | null>(null);

const currentTitle = computed(() =>
  stage.value === 'phone'
    ? 'Жүйеге кіру'
    : 'Кодты растау'
);

const currentDescription = computed(() =>
  stage.value === 'phone'
    ? 'ITutor платформасына кіру үшін телефон нөміріңізді енгізіңіз'
    : `Нөміріне жіберілген 4 сандық кодты енгізіңіз`
);

function handlePhoneSent(phone: string) {
  phoneNumber.value = phone;
  stage.value = 'otp';
}

function handleAuthenticated() {
  router.push('/admin');
}

function handleBackToPhone() {
  stage.value = 'phone';
  phoneNumber.value = null;
}
</script>

<template>
	<div class="space-y-6">
		<!-- Header -->
		<div class="space-y-3 text-center">
			<div class="flex justify-center items-center mb-4">
				<div class="relative">
					<div
						class="flex justify-center items-center bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg rounded-2xl w-16 h-16"
					>
						<Smartphone
							v-if="stage === 'phone'"
							class="w-8 h-8 text-white"
						/>
						<Shield
							v-else
							class="w-8 h-8 text-white"
						/>
					</div>
					<div
						class="-right-1 -bottom-1 absolute flex justify-center items-center bg-gradient-to-br from-green-400 to-emerald-500 shadow-lg rounded-full w-6 h-6"
					>
						<CheckCircle class="w-3 h-3 text-white" />
					</div>
				</div>
			</div>

			<h1 class="font-bold text-gray-900 text-2xl">
				{{ currentTitle }}
			</h1>
			<p class="mx-auto max-w-sm text-gray-600 text-sm leading-relaxed">
				{{ currentDescription }}
			</p>
		</div>

		<!-- Progress Indicator -->
		<div class="flex justify-center items-center space-x-2">
			<div class="flex items-center space-x-2">
				<div
					:class="[
          'w-8 h-2 rounded-full transition-all duration-300',
          stage === 'phone' ? 'bg-gradient-to-r from-blue-500 to-indigo-600' : 'bg-gradient-to-r from-green-400 to-emerald-500'
        ]"
				></div>
				<div
					:class="[
          'w-8 h-2 rounded-full transition-all duration-300',
          stage === 'otp' ? 'bg-gradient-to-r from-blue-500 to-indigo-600' : 'bg-gray-200'
        ]"
				></div>
			</div>
		</div>

		<!-- Form Content -->
		<div class="space-y-6">
			<Transition
				name="slide"
				mode="out-in"
			>
				<PhoneLoginForm
					v-if="stage === 'phone'"
					key="phone"
					@sent="handlePhoneSent"
				/>
				<div
					v-else
					key="otp"
					class="space-y-4"
				>
					<OtpLoginForm
						:phone-number="phoneNumber as string"
						@authenticated="handleAuthenticated"
						@back-to-phone="handleBackToPhone"
					/>

					<!-- Back Button -->
					<button
						@click="handleBackToPhone"
						class="flex justify-center items-center gap-2 hover:bg-gray-50 p-3 border border-gray-200 hover:border-gray-300 rounded-xl w-full text-gray-600 hover:text-gray-800 transition-all duration-200"
					>
						<ArrowLeft class="w-4 h-4" />
						<span class="font-medium">Артқа қайту</span>
					</button>
				</div>
			</Transition>
		</div>
	</div>
</template>

<style scoped>
.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease-out;
}

.slide-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.slide-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}
</style>
