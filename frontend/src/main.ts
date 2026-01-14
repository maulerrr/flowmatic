import App from '@/App.vue'
import router from '@/router'
import '@/styles.css'
import { VueQueryPlugin } from '@tanstack/vue-query'
import { createPinia } from 'pinia'
import { createApp } from 'vue'

import { apiClient } from '@/core/configs/axios-instance.config'
import { vueQueryPluginOptions } from '@/core/configs/query-client.config'

// Restore persisted tokens (if any) to keep session after closing tab
try {
	const raw = localStorage.getItem('authTokens')
	if (raw) {
		const parsed = JSON.parse(raw)
		if (parsed?.accessToken) {
			apiClient.defaults.headers.common['Authorization'] = `Bearer ${parsed.accessToken}`
		}
	}
} catch (e) {
	console.warn('Failed to restore auth tokens', e)
}

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(VueQueryPlugin, vueQueryPluginOptions)

app.mount('#app')
