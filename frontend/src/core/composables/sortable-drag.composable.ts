// src/modules/kiosk/products/composables/useSortableList.tsAdd commentMore actions
import Sortable, { type SortableOptions } from 'sortablejs'
import { type Ref, nextTick, onBeforeUnmount, onMounted, watch } from 'vue'

/**
 * Reusable Sortable.js wrapper.
 *
 * @param listRef      — ref to the <ul> (or other) container
 * @param items        — reactive array of items to reorder
 * @param onReorder    — callback(newList) when order changes
 * @param options      — any SortableOptions overrides
 */
export function useSortableList<T>(
	listRef: Ref<HTMLElement | null>,
	items: Ref<T[]>,
	onReorder: (newList: T[]) => void,
	options: Partial<SortableOptions> = {},
) {
	let sortable: Sortable | null = null

	const init = async () => {
		await nextTick()
		if (sortable) {
			sortable.destroy()
			sortable = null
		}
		if (!listRef.value) return

		sortable = Sortable.create(listRef.value, {
			animation: 200,
			handle: options.handle ?? '.drag-handle',
			ghostClass: 'sortable-ghost',
			chosenClass: 'sortable-chosen',
			onEnd(evt) {
				const { oldIndex, newIndex } = evt
				if (oldIndex == null || newIndex == null) return

				const updated = [...items.value]
				const [moved] = updated.splice(oldIndex, 1)
				updated.splice(newIndex, 0, moved)
				onReorder(updated)
			},
			...options,
		})
	}

	onMounted(init)
	watch(items, init, { deep: true })
	onBeforeUnmount(() => sortable?.destroy())
}
