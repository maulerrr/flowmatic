import { DataRow } from 'src/common/types/data.types'

export interface ParseCsvResult {
	rows: DataRow[]
	columns: string[]
}

export function parseCsvBuffer(buffer: Buffer): ParseCsvResult {
	const text = buffer.toString('utf-8').trim()
	if (!text) return { rows: [], columns: [] }

	const lines = text.split(/\r?\n/).filter(line => line.length > 0)
	if (lines.length === 0) return { rows: [], columns: [] }

	const columns = parseCsvLine(lines[0]).map(column => column.trim())
	const rows: DataRow[] = []

	for (let i = 1; i < lines.length; i++) {
		const values = parseCsvLine(lines[i])
		if (values.length === 0) continue

		const row: DataRow = {}
		columns.forEach((column, index) => {
			row[column] = values[index] ?? ''
		})
		rows.push(row)
	}

	return { rows, columns }
}

export function parseCsvLine(line: string): string[] {
	const result: string[] = []
	let current = ''
	let inQuotes = false

	for (let i = 0; i < line.length; i++) {
		const char = line[i]

		if (char === '"') {
			if (inQuotes && line[i + 1] === '"') {
				current += '"'
				i++
			} else {
				inQuotes = !inQuotes
			}
		} else if (char === ',' && !inQuotes) {
			result.push(current)
			current = ''
		} else {
			current += char
		}
	}

	result.push(current)
	return result
}

export function escapeCsvValue(value: string, delimiter: string = ','): string {
	if (value.includes(delimiter) || value.includes('"') || value.includes('\n')) {
		return `"${value.replace(/"/g, '""')}"`.replace(/\n/g, '\\n')
	}
	return value
}

export function rowsToCsv(
	data: Record<string, unknown>[],
	options?: { delimiter?: string; headers?: string[] },
): string {
	if (data.length === 0) return ''

	const delimiter = options?.delimiter ?? ','
	const headers = options?.headers ?? Object.keys(data[0])
	const headerLine = headers.map(header => escapeCsvValue(header, delimiter)).join(delimiter)
	const dataLines = data.map(row =>
		headers
			.map(header => {
				const value = row[header]
				if (value === null || value === undefined) return ''
				if (typeof value === 'object') return escapeCsvValue(JSON.stringify(value), delimiter)
				return escapeCsvValue(String(value), delimiter)
			})
			.join(delimiter),
	)

	return [headerLine, ...dataLines].join('\n')
}
