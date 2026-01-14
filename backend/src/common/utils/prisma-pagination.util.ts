import {
	DEFAULT_PAGINATION,
	PaginatedResponse,
	PaginationParamsFilter,
	createPaginationMeta,
} from './pagination.util'

/**
 * Generic page-based pagination helper for any Prisma model.
 *
 * @param delegate          Prisma model delegate, e.g. prisma.module
 * @param findArgs          all the usual Prisma findMany args except skip/take
 * @param countDelegate     Prisma model delegate used for counting (usually same model)
 * @param countArgs         all the usual Prisma count args (usually just { where })
 * @param options.page      current page (1-based)
 * @param options.pageSize  items per page
 * @param options.disablePagination  if true, return all records (no skip/take)
 */
export async function paginatePrisma<T, F extends object>(
	delegate: { findMany(args: F & { skip?: number; take?: number }): Promise<T[]> },
	findArgs: Omit<F, 'skip' | 'take'>,
	countDelegate: { count(args: F): Promise<number> },
	countArgs: F,
	options?: PaginationParamsFilter,
): Promise<PaginatedResponse<T>> {
	const {
		page = DEFAULT_PAGINATION.page,
		pageSize = DEFAULT_PAGINATION.pageSize,
		disablePagination,
	} = options ?? {}

	if (disablePagination) {
		const data = await delegate.findMany(findArgs as F & { skip?: number; take?: number })
		const totalCount = data.length

		return {
			data,
			pagination: createPaginationMeta(1, totalCount, totalCount),
		}
	}

	const skip = (page - 1) * pageSize
	const [data, totalCount] = await Promise.all([
		delegate.findMany({ ...findArgs, skip, take: pageSize } as F & { skip: number; take: number }),
		countDelegate.count(countArgs),
	])

	return {
		data,
		pagination: createPaginationMeta(page, pageSize, totalCount),
	}
}
