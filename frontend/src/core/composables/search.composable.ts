import { ref } from 'vue'

import type { UseSearchOptions } from '../models/search.model'

export function useSearch(options: UseSearchOptions = {}) {
	const { initialValue = '', debounceTime = 400, onSearch } = options

	const searchInput = ref(initialValue)
	const searchTimeout = ref<number | null>(null)
	const isSearching = ref(false)

	const handleSearch = (value: string) => {
		searchInput.value = value
		isSearching.value = true

		if (searchTimeout.value) {
			clearTimeout(searchTimeout.value)
		}

		searchTimeout.value = setTimeout(() => {
			if (onSearch) {
				onSearch(value)
			}
			isSearching.value = false
		}, debounceTime) as unknown as number
	}

	const clearSearch = () => {
		searchInput.value = ''
		if (onSearch) {
			onSearch('')
		}
	}

	const cleanup = () => {
		if (searchTimeout.value) {
			clearTimeout(searchTimeout.value)
		}
	}

	return {
		searchInput,
		isSearching,
		handleSearch,
		clearSearch,
		cleanup,
	}
}
