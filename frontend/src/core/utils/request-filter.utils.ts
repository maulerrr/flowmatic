/* eslint-disable @typescript-eslint/no-explicit-any */
type FilterObject<T> = {
	[K in keyof T]: T[K] extends object
		? FilterObject<T[K]> // Recursively filter nested objects
		: T[K]
}

const isValidValue = (value: unknown): boolean => {
	// Check if the value is valid (not null, undefined, or empty string)
	return (
		value !== null && value !== undefined && !(typeof value === 'string' && value.trim() === '')
	)
}

const filterObject = (input: any): any => {
	return Object.entries(input).reduce((acc, [key, value]) => {
		if (isValidValue(value)) {
			// Recursively filter if the value is an object
			acc[key] = typeof value === 'object' && !Array.isArray(value) ? filterObject(value) : value
		}
		return acc
	}, {} as any)
}

/**
 * Removes all null, undefined, empty string, and invalid values from an object.
 * @param obj - The input object to be filtered.
 * @returns A new object with invalid fields removed.
 */
export const buildRequestFilter = <T extends object>(obj: T | undefined): FilterObject<T> => {
	if (!obj) return {} as any
	return filterObject(obj) as FilterObject<T>
}
