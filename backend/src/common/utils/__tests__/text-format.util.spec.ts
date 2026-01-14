import { stripHtmlToPlainTextSimple } from '../text-format.util'

describe('stripHtmlToPlainTextSimple', () => {
	it('removes newline after opening quotes and before closing quotes', () => {
		const input = 'Мысалы: «Сәлем,\n!»'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toContain('Мысалы: «Сәлем!»')
	})

	it('replaces comma+newline with space', () => {
		const input = 'Hello,\nworld!'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toBe('Hello, world!')
	})

	it('removes extra spaces around punctuation near quotes', () => {
		const input = '« Сәлем , әлем ! »'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toBe('«Сәлем, әлем!»')
	})

	it('joins lone punctuation on separate lines with the surrounding text', () => {
		const input = 'Сәлем,\n!\n- «Сәлеметсің бе,\n?\n»'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toContain('Сәлем!')
		expect(out).toContain('«Сәлеметсің бе?»')
	})

	it('converts simple HTML lists to dash bullets and strips tags', () => {
		const input = '<ul>\n<li>One</li>\n<li>Two</li>\n</ul>'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toContain('- One')
		expect(out).toContain('- Two')
		expect(out).not.toMatch(/<li>|<ul>/)
	})

	it('decodes basic HTML entities', () => {
		const input = 'Fish &amp; Chips — 5 &lt; 6 &gt; 4'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toBe('Fish & Chips — 5 < 6 > 4')
	})

	it('does not strip non-HTML angle-bracketed content and unwraps kz-t markers', () => {
		const input = 'Use <angle> like this and <kz-t>Сәлем</kz-t> remains Сәлем'
		const out = stripHtmlToPlainTextSimple(input)
		expect(out).toContain('Use <angle> like this')
		expect(out).toContain('Сәлем')
		expect(out).not.toMatch(new RegExp('<kz-t>|</kz-t>', 'i'))
	})
})
