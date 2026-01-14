import {
	QueryClient,
	type QueryKey,
	type UseQueryOptions,
	type VueQueryPluginOptions,
} from '@tanstack/vue-query'

/**
 * A base type for react-query hooks that removes
 * `queryKey` and `queryFn` from the usual options,
 * so callers only need to supply the things they want to override.
 *
 * @template TQueryFnData  The raw data returned by your fetcher
 * @template TData         The data shape after any select/mapping (defaults to TQueryFnData)
 * @template TError        The error type (defaults to Error)
 */
export type BaseQueryOptions<TQueryFnData, TData = TQueryFnData, TError = Error> = Omit<
	UseQueryOptions<TQueryFnData, TError, TData, QueryKey>,
	'queryKey' | 'queryFn'
>

export const queryClient = new QueryClient({
	// Global default options for all queries and mutations
	defaultOptions: {
		queries: {
			staleTime: 1000 * 60, // 1 minute
			gcTime: 1000 * 60 * 5, // 5 minutes
			refetchOnWindowFocus: true, // You can set this to `false` for less critical data
			refetchOnMount: true, // Refetch when a component mounts if the data is stale
			refetchOnReconnect: true, // Refetch when the network connection is re-established
			retry: 3,
			retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
		},
		mutations: {
			retry: 0,
		},
	},
})

// The VueQueryPluginOptions object to pass to app.use(VueQueryPlugin, vueQueryPluginOptions)
export const vueQueryPluginOptions: VueQueryPluginOptions = {
	queryClient, // Pass the created QueryClient instance
}
