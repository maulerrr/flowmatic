export class PaginationParamsFilter {
	page?: number
	pageSize?: number
	disablePagination?: boolean
}

export interface PaginationMeta {
	page: number
	pageSize: number
	totalCount: number
	totalPages: number
}

export interface PaginatedResponse<T> {
	data: T[]
	pagination: PaginationMeta
}
