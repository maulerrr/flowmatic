import { Injectable, Logger } from '@nestjs/common'
import { DataRow } from 'src/common/types/data.types'

export interface QualityReport {
	missing: Record<string, number>
	duplicates: number
	outliers: {
		count: number
		rows: DataRow[]
		columns: string[]
	}
	totalRows: number
	totalColumns: number
	numericColumns: string[]
	categoricalColumns: string[]
	summary: {
		missingPercentage: number
		duplicatePercentage: number
		outlierPercentage: number
	}
}

@Injectable()
export class QualityService {
	private readonly logger = new Logger(QualityService.name)

	analyzeQuality(data: DataRow[], columns: string[]): QualityReport {
		const missing = this.reportMissing(data, columns)
		const duplicates = this.reportDuplicates(data)
		const { numericColumns, categoricalColumns } = this.classifyColumns(data, columns)
		const outliers = this.detectOutliers(data, numericColumns)

		const totalRows = data.length
		const totalColumns = columns.length
		const totalMissing = Object.values(missing).reduce((a, b) => a + b, 0)

		return {
			missing,
			duplicates,
			outliers,
			totalRows,
			totalColumns,
			numericColumns,
			categoricalColumns,
			summary: {
				missingPercentage: (totalMissing / (totalRows * totalColumns)) * 100,
				duplicatePercentage: (duplicates / totalRows) * 100,
				outlierPercentage: (outliers.count / totalRows) * 100,
			},
		}
	}

	private reportMissing(data: DataRow[], columns: string[]): Record<string, number> {
		const missing: Record<string, number> = {}

		columns.forEach(col => {
			missing[col] = data.filter(
				row => row[col] === null || row[col] === undefined || row[col] === '',
			).length
		})

		return missing
	}

	private reportDuplicates(data: DataRow[]): number {
		const seen = new Set<string>()
		let duplicateCount = 0

		for (const row of data) {
			const key = JSON.stringify(row)
			if (seen.has(key)) {
				duplicateCount++
			} else {
				seen.add(key)
			}
		}

		return duplicateCount
	}

	private classifyColumns(
		data: DataRow[],
		columns: string[],
	): { numericColumns: string[]; categoricalColumns: string[] } {
		const numericColumns: string[] = []
		const categoricalColumns: string[] = []

		columns.forEach(col => {
			const sample = data
				.slice(0, Math.min(100, data.length))
				.map(row => row[col])
				.filter(val => val !== null && val !== undefined && val !== '')

			if (sample.length === 0) {
				categoricalColumns.push(col)
				return
			}

			const numericCount = sample.filter(
				val => typeof val === 'number' || !isNaN(Number(val)),
			).length

			if (numericCount / sample.length > 0.8) {
				numericColumns.push(col)
			} else {
				categoricalColumns.push(col)
			}
		})

		return { numericColumns, categoricalColumns }
	}

	private detectOutliers(
		data: DataRow[],
		numericColumns: string[],
		threshold: number = 3,
	): { count: number; rows: DataRow[]; columns: string[] } {
		// Pre-compute stats per column once (O(n*m)) instead of per row (O(n^2*m))
		const stats = new Map<string, { mean: number; stdDev: number }>()
		for (const col of numericColumns) {
			const values = data.map(r => Number(r[col])).filter(v => !isNaN(v))
			if (values.length < 2) continue
			const mean = values.reduce((a, b) => a + b, 0) / values.length
			const variance = values.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / values.length
			const stdDev = Math.sqrt(variance)
			if (stdDev > 0) stats.set(col, { mean, stdDev })
		}

		const outlierRows: DataRow[] = []
		const outlierColumns = new Set<string>()

		for (const row of data) {
			let isOutlier = false
			for (const col of stats.keys()) {
				const value = Number(row[col])
				if (isNaN(value)) continue
				const { mean, stdDev } = stats.get(col)!
				if (Math.abs((value - mean) / stdDev) > threshold) {
					isOutlier = true
					outlierColumns.add(col)
				}
			}
			if (isOutlier) outlierRows.push(row)
		}

		return {
			count: outlierRows.length,
			rows: outlierRows,
			columns: Array.from(outlierColumns),
		}
	}
}
