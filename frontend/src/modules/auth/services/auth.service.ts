import { type AxiosError, isAxiosError } from 'axios'

import { apiClient } from '@/core/configs/axios-instance.config'

import type { AdminDto } from '../models/auth.models'

export class AuthServiceError extends Error {
	constructor(
		message: string,
		public statusCode?: number,
	) {
		super(message)
		this.name = 'AuthServiceError'
		Object.setPrototypeOf(this, AuthServiceError.prototype)
	}
}

class AuthService {
	async getCurrentUser() {
		try {
			const response = await apiClient.get<AdminDto>('/admin/me')
			if (response.status >= 200 && response.status < 300) {
				return response.data
			} else {
				throw new AuthServiceError(
					`Failed to fetch current user. Status: ${response.status}`,
					response.status,
				)
			}
		} catch (error) {
			if (isAxiosError(error)) {
				const axiosError = error as AxiosError
				throw new AuthServiceError(
					axiosError.message || 'Failed to fetch current user',
					axiosError.response?.status,
				)
			} else {
				throw new AuthServiceError('Unknown error', 500)
			}
		}
	}

	async login(email: string, password: string) {
		try {
			const response = await apiClient.post('/auth/login', { email, password })
			if (response.status >= 200 && response.status < 300) {
				const { token } = response.data
				this.setToken(token)
				return response.data
			} else {
				throw new AuthServiceError(`Login failed. Status: ${response.status}`, response.status)
			}
		} catch (error) {
			if (isAxiosError(error)) {
				const axiosError = error as AxiosError
				throw new AuthServiceError(
					axiosError.message || 'Login failed',
					axiosError.response?.status,
				)
			} else {
				throw new AuthServiceError('Unknown error', 500)
			}
		}
	}

	async logout() {
		try {
			await apiClient.post('/auth/logout')
			this.clearToken()
		} catch (error) {
			if (isAxiosError(error)) {
				const axiosError = error as AxiosError
				throw new AuthServiceError(
					axiosError.message || 'Logout failed',
					axiosError.response?.status,
				)
			} else {
				throw new AuthServiceError('Unknown error', 500)
			}
		}
	}

	isAuthenticated(): boolean {
		if (typeof window === 'undefined') return false
		return !!localStorage.getItem('auth_token')
	}

	getToken(): string | null {
		if (typeof window === 'undefined') return null
		return localStorage.getItem('auth_token')
	}

	setToken(token: string): void {
		if (typeof window !== 'undefined') {
			localStorage.setItem('auth_token', token)
		}
	}

	clearToken(): void {
		if (typeof window !== 'undefined') {
			localStorage.removeItem('auth_token')
		}
	}

	async sendOtp(phoneNumber: string): Promise<void> {
		// Stub implementation for OTP flow
		console.log(`OTP sent to ${phoneNumber}`)
	}

	async verifyOtp(phoneNumber: string, code: string): Promise<void> {
		// Stub implementation for OTP verification
		this.setToken(`token_${code}`)
		console.log(`OTP verified for ${phoneNumber}`)
	}
}

export const authService = new AuthService()
