import { IsOptional, IsInt, Min, IsBoolean } from 'class-validator'
import { Transform, Type } from 'class-transformer'

export const DEFAULT_PAGINATION = {
	page: 1,
	pageSize: 10,
}

export class PaginationParamsFilter {
	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	page?: number = DEFAULT_PAGINATION.page

	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	pageSize?: number = DEFAULT_PAGINATION.pageSize

	@IsOptional()
	@Transform(({ value }) => {
		// only treat exactly "true" (string or boolean) as true
		if (value === 'true' || value === true) return true
		return false
	})
	@IsBoolean()
	disablePagination?: boolean = false
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

export function createPaginationMeta(
	page: number,
	pageSize: number,
	totalCount: number,
): PaginationMeta {
	const totalPages = Math.ceil(totalCount / pageSize) || 1
	return { page, pageSize, totalCount, totalPages }
}

export function hasMorePages(meta: PaginationMeta): boolean {
	return meta.page < meta.totalPages
}

/**
 * Wrap raw data + meta into a PaginatedResponse
 */
export function buildPaginatedResponse<T>(
	data: T[],
	pagination: PaginationMeta,
): PaginatedResponse<T> {
	return { data, pagination }
}

/**
 * Map over a PaginatedResponse's items, carrying the same pagination meta
 */
export function mapPaginatedResponse<In, Out>(
	page: PaginatedResponse<In>,
	fn: (item: In) => Out,
): PaginatedResponse<Out> {
	return {
		data: page.data.map(fn),
		pagination: page.pagination,
	}
}
