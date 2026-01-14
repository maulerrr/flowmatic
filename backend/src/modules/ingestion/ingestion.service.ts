import { Injectable, Logger } from '@nestjs/common'
import * as fs from 'fs/promises'
import { createReadStream } from 'fs'
import * as path from 'path'

export interface DataRow {
	[key: string]: any
}

export interface IngestedData {
	data: DataRow[]
	columns: string[]
	rowCount: number
	columnCount: number
	source: string
	sourceType: string
	detectedDatetimeColumn?: string
}

@Injectable()
export class IngestionService {
	private readonly logger = new Logger(IngestionService.name)

	async ingestFromFile(filePath: string): Promise<IngestedData> {
		const ext = path.extname(filePath).toLowerCase()

		if (ext === '.csv') {
			return this.ingestCSV(filePath)
		} else if (ext === '.json') {
			return this.ingestJSON(filePath)
		} else {
			throw new Error(`Unsupported file format: ${ext}`)
		}
	}

	private async ingestCSV(filePath: string): Promise<IngestedData> {
		const content = await fs.readFile(filePath, 'utf-8')
		const lines = content.trim().split('\n')

		if (lines.length < 1) {
			throw new Error('CSV file is empty')
		}

		const headers = lines[0].split(',').map(h => h.trim())
		const data: DataRow[] = []

		for (let i = 1; i < lines.length; i++) {
			const values = lines[i].split(',').map(v => {
				const trimmed = v.trim()
				// Try to parse as number
				const num = Number(trimmed)
				return isNaN(num) ? trimmed : num
			})
			const row: DataRow = {}
			headers.forEach((header, idx) => {
				row[header] = values[idx]
			})
			data.push(row)
		}

		const datetimeColumn = this.detectDatetimeColumn(data, headers)

		return {
			data,
			columns: headers,
			rowCount: data.length,
			columnCount: headers.length,
			source: filePath,
			sourceType: 'csv',
			detectedDatetimeColumn: datetimeColumn,
		}
	}

	private async ingestJSON(filePath: string): Promise<IngestedData> {
		const content = await fs.readFile(filePath, 'utf-8')
		const json = JSON.parse(content)

		let data: DataRow[]
		if (Array.isArray(json)) {
			data = json
		} else if (json.data && Array.isArray(json.data)) {
			data = json.data
		} else {
			throw new Error('JSON must be an array or have a "data" array property')
		}

		const columns = data.length > 0 ? Object.keys(data[0]) : []
		const datetimeColumn = this.detectDatetimeColumn(data, columns)

		return {
			data,
			columns,
			rowCount: data.length,
			columnCount: columns.length,
			source: filePath,
			sourceType: 'json',
			detectedDatetimeColumn: datetimeColumn,
		}
	}

	private detectDatetimeColumn(data: DataRow[], columns: string[]): string | undefined {
		const datetimeKeywords = ['date', 'time', 'timestamp', 'datetime', 'created', 'updated']

		for (const col of columns) {
			const lowerCol = col.toLowerCase()
			if (datetimeKeywords.some(keyword => lowerCol.includes(keyword))) {
				if (data.length > 0 && data[0][col]) {
					const parsed = new Date(data[0][col])
					if (!isNaN(parsed.getTime())) {
						return col
					}
				}
			}
		}

		return undefined
	}
}
