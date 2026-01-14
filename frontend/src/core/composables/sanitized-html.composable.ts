// composables/useSanitizedHtml.ts
import DOMPurify from 'dompurify'
import { computed } from 'vue'

/**
 * Returns a computed ref of sanitized HTML.
 */
export function useSanitizedHtml(rawHtml: string | null | undefined) {
	return computed(() => (rawHtml ? DOMPurify.sanitize(rawHtml) : ''))
}
