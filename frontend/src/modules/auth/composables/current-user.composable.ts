import { useQuery } from '@tanstack/vue-query'

import type { BaseQueryOptions } from '@/core/configs/query-client.config'

import type { AdminDto } from '../models/auth.models'
import { authService } from '../services/auth.service'

export const CURRENT_USER_QUERY_KEY = 'current-user'

export const useCurrentUser = (queryOptions?: BaseQueryOptions<AdminDto, AdminDto>) => {
	return useQuery({
		...queryOptions,
		queryKey: [CURRENT_USER_QUERY_KEY],
		queryFn: () => authService.getCurrentUser(),
	})
}
