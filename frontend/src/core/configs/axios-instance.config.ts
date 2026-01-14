import axios from 'axios'

const REQ_TIMEOUT = 60000

const apiUrl = `${import.meta.env.VITE_API_URL || 'http://example:8080/api/v1'}`

export const apiClient = axios.create({
	baseURL: apiUrl,
	timeout: REQ_TIMEOUT,
	withCredentials: true,
})

apiClient.interceptors.request.use(
	config => {
		return config
	},
	error => {
		return Promise.reject(error)
	},
)

apiClient.interceptors.response.use(
	response => response,
	async error => {
		if (error.response && error.response.status === 401) {
			console.error('Unauthorized access - redirecting to login')
		}
		return Promise.reject(error)
	},
)
