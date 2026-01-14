<script setup lang="ts">
import { Pagination, PaginationContent, PaginationEllipsis, PaginationItem, PaginationNext, PaginationPrevious } from '@/core/components/ui/pagination';





defineProps<{
    totalPages: number,
    defaultPage: number,
    pageSize: number,
    totalCount: number,
}>()

const emit = defineEmits<{
	(e: 'page-change', page: number): void;
}>();

const changePage = (page: number) => {
  emit('page-change', page)
}
</script>

<template>
	<div
		class="flex items-center justify-center space-x-2 pt-4"
		v-if="totalPages > 1"
	>
		<Pagination
			v-slot="{ page }"
			:items-per-page="pageSize"
			:total="totalCount"
			:default-page="defaultPage"
		>
			<PaginationContent>
				<PaginationPrevious
					:disabled="page <= 1"
					@click="changePage(page - 1)"
				/>

				<PaginationItem
					:value="1"
					:is-active="page === 1"
					@click="changePage(1)"
				>
					1
				</PaginationItem>

				<PaginationEllipsis v-if="page > 3" />

				<PaginationItem
					v-if="page > 2"
					:value="page - 1"
					:is-active="false"
					@click="changePage(page - 1)"
				>
					{{ page - 1 }}
				</PaginationItem>

				<PaginationItem
					v-if="page > 1 && page < totalPages"
					:value="page"
					:is-active="true"
				>
					{{ page }}
				</PaginationItem>

				<PaginationItem
					v-if="page < totalPages - 1"
					:value="page + 1"
					:is-active="false"
					@click="changePage(page + 1)"
				>
					{{ page + 1 }}
				</PaginationItem>

				<PaginationEllipsis v-if="page < totalPages - 2" />

				<PaginationItem
					v-if="totalPages > 1"
					:value="totalPages"
					:is-active="page === totalPages"
					@click="changePage(totalPages)"
				>
					{{ totalPages }}
				</PaginationItem>

				<PaginationNext
					:disabled="page >= totalPages"
					@click="changePage(page + 1)"
				/>
			</PaginationContent>
		</Pagination>
	</div>
</template>
