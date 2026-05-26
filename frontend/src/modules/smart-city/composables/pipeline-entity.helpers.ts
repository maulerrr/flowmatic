export function replaceInList<T extends { id: string }>(list: T[], item?: T | null): void {
	if (!item) return
	const index = list.findIndex(entry => entry.id === item.id)
	if (index >= 0) list[index] = item
}

export function removeFromList<T extends { id: string }>(list: T[], id: string): T[] {
	return list.filter(entry => entry.id !== id)
}
