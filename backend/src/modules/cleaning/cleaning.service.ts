import { Injectable, Logger } from '@nestjs/common'
import { DataRow } from '../ingestion/ingestion.service'

export interface CleaningResult {
	data: DataRow[]
	changes: {
		duplicatesRemoved: number
		missingImputed: number
		outliersHandled: number
	}
}

@Injectable()
export class CleaningService {
	private readonly logger = new Logger(CleaningService.name)

	clean(data: DataRow[], numericColumns: string[], categoricalColumns: string[]): CleaningResult {
		let cleanedData = [...data]
		const changes = {
			duplicatesRemoved: 0,
			missingImputed: 0,
			outliersHandled: 0,
		}

		// Step 1: Remove duplicates
		const { data: deduped, removed } = this.removeDuplicates(cleanedData)
		cleanedData = deduped
		changes.duplicatesRemoved = removed

		// Step 2: Impute missing values
		const { data: imputed, count: imputedCount } = this.imputeMissing(
			cleanedData,
			numericColumns,
			categoricalColumns,
		)
		cleanedData = imputed
		changes.missingImputed = imputedCount

		// Step 3: Handle outliers
		const { data: capped, count: cappedCount } = this.handleOutliers(cleanedData, numericColumns)
		cleanedData = capped
		changes.outliersHandled = cappedCount

		return {
			data: cleanedData,
			changes,
		}
	}

	private removeDuplicates(data: DataRow[]): { data: DataRow[]; removed: number } {
		const seen = new Set<string>()
		const unique: DataRow[] = []

		for (const row of data) {
			const key = JSON.stringify(row)
			if (!seen.has(key)) {
				seen.add(key)
				unique.push(row)
			}
		}

		return {
			data: unique,
			removed: data.length - unique.length,
		}
	}

	private imputeMissing(
		data: DataRow[],
		numericColumns: string[],
		categoricalColumns: string[],
	): { data: DataRow[]; count: number } {
		let imputedCount = 0
		const result: DataRow[] = data.map(row => ({ ...row }))

		// Impute numeric columns with mean
		numericColumns.forEach(col => {
			const values = data
				.map(r => Number(r[col]))
				.filter(v => !isNaN(v) && v !== null && v !== undefined)

			if (values.length === 0) return

			const mean = values.reduce((a, b) => a + b, 0) / values.length

			result.forEach(row => {
				if (
					row[col] === null ||
					row[col] === undefined ||
					row[col] === '' ||
					isNaN(Number(row[col]))
				) {
					row[col] = mean
					imputedCount++
				}
			})
		})

		// Impute categorical columns with mode
		categoricalColumns.forEach(col => {
			const values = data.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '')

			if (values.length === 0) return

			// Find mode
			const freq: Record<string, number> = {}
			values.forEach(v => {
				freq[String(v)] = (freq[String(v)] || 0) + 1
			})
			const mode = Object.keys(freq).reduce((a: string, b: string) => (freq[a] > freq[b] ? a : b))

			result.forEach(row => {
				if (row[col] === null || row[col] === undefined || row[col] === '') {
					row[col] = mode
					imputedCount++
				}
			})
		})

		return { data: result, count: imputedCount } as { data: DataRow[]; count: number }
	}

	private handleOutliers(
		data: DataRow[],
		numericColumns: string[],
		threshold: number = 3,
	): { data: DataRow[]; count: number } {
		let cappedCount = 0
		const result: DataRow[] = data.map(row => ({ ...row }))

		numericColumns.forEach(col => {
			const values = data
				.map(r => Number(r[col]))
				.filter(v => !isNaN(v) && v !== null && v !== undefined)

			if (values.length < 2) return

			const mean = values.reduce((a, b) => a + b, 0) / values.length
			const variance = values.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / values.length
			const stdDev = Math.sqrt(variance)

			if (stdDev === 0) return

			const lowerBound = mean - threshold * stdDev
			const upperBound = mean + threshold * stdDev

			result.forEach(row => {
				const value = Number(row[col])
				if (isNaN(value)) return

				if (value < lowerBound) {
					row[col] = lowerBound
					cappedCount++
				} else if (value > upperBound) {
					row[col] = upperBound
					cappedCount++
				}
			})
		})

		return { data: result, count: cappedCount }
	}
}
