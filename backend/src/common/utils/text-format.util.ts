export function stripHtmlToPlainTextSimple(input: string): string {
	if (!input) return ''
	let text = String(input)

	// Remove Markdown code fences if present
	text = text
		.trim()
		.replace(/^```(?:json)?\s*/i, '')
		.replace(/\s*```$/i, '')

	// Normalize common HTML block separators to newlines
	text = text
		.replace(/<\s*br\s*\/?>/gi, '\n')
		.replace(/<\s*\/p\s*>/gi, '\n\n')
		.replace(/<\s*p\b[^>]*>/gi, '')

	// Convert list items to dash bullets; handle ordered/unordered lists
	text = text
		.replace(/<\s*li\b[^>]*>/gi, '\n- ')
		.replace(/<\s*\/li\s*>/gi, '')
		.replace(/<\s*ul\b[^>]*>/gi, '')
		.replace(/<\s*\/ul\s*>/gi, '\n')
		.replace(/<\s*ol\b[^>]*>/gi, '')
		.replace(/<\s*\/ol\s*>/gi, '\n')

	// Strip any remaining tags
	// First, remove our own protected-term wrappers explicitly
	text = text.replace(/<\/?kz-t>/gi, '')
	// Then, strip only known formatting tags to avoid deleting content that uses angle brackets literally
	// Ensure we match exact tag names (e.g., <a ...> but not <angle>) using a lookahead for space or '>'
	text = text.replace(/<\/?(strong|em|b|i|u|code|span|div|a)(?=\s|>)[^>]*>/gi, '')

	// Decode a few common entities
	text = text
		.replace(/&nbsp;/g, ' ')
		.replace(/&amp;/g, '&')
		.replace(/&lt;/g, '<')
		.replace(/&gt;/g, '>')
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")

	// Tidy broken line breaks around punctuation and quotes (common LLM/HTML strip artifacts)
	// 1) Comma/semicolon/colon followed by a newline → replace with a space
	text = text.replace(/([,;:])\s*\n\s*/g, '$1 ')
	// 2) Newline right before closing punctuation or closing quotes → remove the newline
	text = text.replace(/\n\s*([»”"'!)?.,;:])/g, '$1')
	// 3) Newline right after opening quotes or opening parenthesis → remove the newline
	text = text.replace(/([«“"'(])\s*\n\s*/g, '$1')
	// 4) Remove extra spaces before closing punctuation/quotes
	text = text.replace(/\s+([»”"'!?.,;:])/g, '$1')
	// 5) Remove extra spaces immediately after opening quotes/parenthesis
	text = text.replace(/([«“"'(])\s+/g, '$1')
	// 6) Remove redundant comma before ! or ? (common artifact when line breaks are fixed)
	text = text.replace(/,\s*([!?])/g, '$1')

	// Collapse excessive whitespace and normalize blank lines
	text = text
		.replace(/\r\n|\r/g, '\n')
		.replace(/[\t\f\v]/g, ' ')
		.replace(/[ \u00A0]+/g, ' ')
		.replace(/\n{3,}/g, '\n\n')
		.trim()

	return text
}

export function wrapProtectedTerms(
	input: string,
	terms: string[],
	start: string = '',
	end: string = '',
): string {
	if (!input || !terms?.length) return input || ''
	let out = input

	// Sort by length desc to prefer longer phrases first
	const unique = Array.from(new Set(terms.filter(t => t && t.trim().length > 0)))
		.map(t => t.trim())
		.sort((a, b) => b.length - a.length)

	for (const term of unique) {
		const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
		// Unicode-aware non-letter boundaries (no lookbehind to keep compatibility)
		const re = new RegExp(`(^|[^\\p{L}\\p{N}_])(${escaped})(?=([^\\p{L}\\p{N}_]|$))`, 'giu')

		out = out.replace(
			re,
			(match, pre: string, core: string, postLook: string, offset: number, whole: string) => {
				const startIdx = offset + (pre ? String(pre).length : 0)
				const endIdx = startIdx + core.length
				// If already wrapped with provided tags, skip
				const left = whole.slice(Math.max(0, startIdx - start.length), startIdx)
				const right = whole.slice(endIdx, endIdx + end.length)
				if (left === start && right === end) {
					return match
				}
				return `${pre ?? ''}${start}${core}${end}`
			},
		)
	}

	return out
}
