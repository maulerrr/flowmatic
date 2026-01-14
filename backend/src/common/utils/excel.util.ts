import * as XLSX from 'xlsx'
import { isImportHeaderKey } from '../i18n/excel-l10n'

const SHEET_NAME = 'itutor'

export function parseExcelBuffer<T = Record<string, unknown>>(buffer: Buffer): T[] {
	const workbook = XLSX.read(buffer, { type: 'buffer' })
	let sheet = workbook.Sheets[SHEET_NAME]
	if (!sheet) {
		for (const name of workbook.SheetNames) {
			const candidate = workbook.Sheets[name]
			if (!candidate) continue
			const rows = XLSX.utils.sheet_to_json(candidate, {
				defval: null,
				range: 0,
				header: 1,
			})
			const headerRow = Array.isArray(rows) && rows.length > 0 ? rows[0] : []
			const hasKnown = Array.isArray(headerRow) && headerRow.some(isImportHeaderKey)
			if (hasKnown) {
				sheet = candidate
				break
			}
		}
	}
	if (!sheet) sheet = workbook.Sheets[workbook.SheetNames[0]] ?? SHEET_NAME
	if (!sheet) return []
	return XLSX.utils.sheet_to_json(sheet, { defval: null })
}
