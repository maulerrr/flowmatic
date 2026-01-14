import { ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import type { PaginationParamsFilter } from '@/core/models/pagination.model'

export function usePaginationFilter<T extends PaginationParamsFilter>(
	initialFilter: T,
	options: { defaultPage?: number; defaultPageSize?: number } = {},
) {
	const router = useRouter()
	const route = useRoute()

	const defaultPage = options.defaultPage ?? 1
	const defaultPageSize = options.defaultPageSize ?? 10

	const initialFilterWithDefaults = {
		...initialFilter,
		page: initialFilter.page ?? defaultPage,
		pageSize: initialFilter.pageSize ?? defaultPageSize,
	} as T

	const filter = ref<T>(initialFilterWithDefaults)

	const changePage = (page: number) => {
		if (page < 1) return
		filter.value = { ...filter.value, page } as T
	}

	const updateFilter = (newFilter: Partial<T>) => {
		const shouldResetPage = Object.keys(newFilter).some(
			key =>
				key !== 'page' &&
				key !== 'pageSize' &&
				newFilter[key as keyof Partial<T>] !== filter.value[key as keyof T],
		)

		filter.value = {
			...filter.value,
			...newFilter,
			page: shouldResetPage ? defaultPage : (newFilter.page ?? filter.value.page),
		} as T
	}

	const resetFilter = () => {
		filter.value = { ...initialFilterWithDefaults } as T
	}

	watch(
		() => filter.value,
		newFilter => {
			router.push({
				query: {
					...route.query,
					page: newFilter.page !== 1 ? newFilter.page?.toString() : undefined,
					search: newFilter.search || undefined,
				},
			})
		},
		{ deep: true },
	)

	return {
		filter,
		changePage,
		updateFilter,
		resetFilter,
	}
}
