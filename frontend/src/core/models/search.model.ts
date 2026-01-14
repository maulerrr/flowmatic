export interface UseSearchOptions {
	initialValue?: string
	debounceTime?: number
	onSearch?: (value: string) => void
}
