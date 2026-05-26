import { apiClient } from '@/api/client'
import { createRouter, createWebHistory } from 'vue-router'

// Import pages
const LoginPage = () => import('@/pages/login-page.vue')
const DashboardPage = () => import('@/pages/dashboard-page.vue')
const UploadPage = () => import('@/modules/auth/pages/upload-page.vue')
const PipelinesPage = () => import('@/modules/auth/pages/pipelines-page.vue')
const AnalyticsPage = () => import('@/pages/analytics-page.vue')
const SettingsPage = () => import('@/pages/settings-page.vue')
const ConnectorsPage = () => import('@/pages/connectors-page.vue')
const PipelineInsightsPage = () => import('@/pages/pipeline-insights-page.vue')

const router = createRouter({
	history: createWebHistory(import.meta.env.BASE_URL),
	scrollBehavior() {
		return { top: 0, behavior: 'smooth' }
	},
	routes: [
		{
			path: '/login',
			name: 'LOGIN',
			component: LoginPage,
			meta: { requiresAuth: false },
		},
		{
			path: '/',
			redirect: '/dashboard',
		},
		{
			path: '/dashboard',
			name: 'DASHBOARD',
			component: DashboardPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/upload',
			name: 'UPLOAD',
			component: UploadPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/connectors',
			name: 'CONNECTORS',
			component: ConnectorsPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/insights',
			name: 'PIPELINE_INSIGHTS',
			component: PipelineInsightsPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/pipelines',
			name: 'PIPELINES',
			component: PipelinesPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/analytics',
			name: 'ANALYTICS',
			component: AnalyticsPage,
			meta: { requiresAuth: true },
		},
		{
			path: '/settings',
			name: 'SETTINGS',
			component: SettingsPage,
			meta: { requiresAuth: true },
		},
		// Catch all - redirect to dashboard
		{
			path: '/:pathMatch(.*)*',
			redirect: '/dashboard',
		},
	],
})

// Navigation guard for authentication
router.beforeEach(async (to, from, next) => {
	// Check if route requires authentication
	if (to.meta.requiresAuth !== true) {
		return next()
	}

	try {
		// Try to get user profile to verify session
		await apiClient.getProfile()
		next()
	} catch (error) {
		console.warn('Auth required but user not logged in:', error)
		// Not authenticated, redirect to login
		next({
			path: '/login',
			query: { redirect: to.fullPath },
		})
	}
})

export default router
