// Common helpers used by Excel utils: normalizeHeader + sheet header detector

import { isVocabHeaderRegex } from './vocab-headers.regex'

// Normalize by lowercasing and removing spaces/punctuation; keep latin/cyrillic & digits
export function normalizeHeader(value: string): string {
	return value
		.trim()
		.toLowerCase()
		.replace(/ё/g, 'е')
		.replace(/[^a-zа-я0-9]+/gi, '')
}

// Smart header detection that now ALSO recognizes vocab headers by regex
export function isImportHeaderKey(header: unknown): boolean {
	if (typeof header !== 'string') return false
	// 1) Vocab regex (KZ/RU/EN) — robust & anchored
	if (isVocabHeaderRegex(header)) return true
	return false
}
