import { Injectable, Logger } from '@nestjs/common'
import * as fs from 'fs/promises'
import * as path from 'path'
import { DataRow } from 'src/common/types/data.types'
import { parseCsvBuffer } from 'src/common/utils/csv-parser.util'

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
		const buffer = await fs.readFile(filePath)
		const { rows: data, columns } = parseCsvBuffer(buffer)

		if (columns.length < 1) {
			throw new Error('CSV file is empty')
		}

		const datetimeColumn = this.detectDatetimeColumn(data, columns)

		return {
			data,
			columns,
			rowCount: data.length,
			columnCount: columns.length,
			source: filePath,
			sourceType: 'csv',
			detectedDatetimeColumn: datetimeColumn,
		}
	}

	private async ingestJSON(filePath: string): Promise<IngestedData> {
		const content = await fs.readFile(filePath, 'utf-8')
		const json = JSON.parse(content) as { data?: unknown[]; [key: string]: unknown }

		let data: DataRow[]
		if (Array.isArray(json)) {
			data = json as DataRow[]
		} else if (json.data && Array.isArray(json.data)) {
			data = json.data as DataRow[]
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
				const val = data[0][col]
				if (data.length > 0 && val) {
					const parsed = new Date(String(val as string | number | Date))
					if (!isNaN(parsed.getTime())) {
						return col
					}
				}
			}
		}

		return undefined
	}
}
