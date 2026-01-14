import { Logger } from '@nestjs/common'
import { ExportAdapter, ExportAdapterType, ExportConfig, ExportResult } from '../types/export.types'

/**
 * Abstract base class for all export adapters
 * Provides common functionality and enforces interface contract
 */
export abstract class BaseExportAdapter implements ExportAdapter {
	protected logger = new Logger(this.constructor.name)

	abstract type: ExportAdapterType
	abstract name: string
	abstract description: string
	abstract requiredSettings: string[]

	/**
	 * Validate adapter-specific configuration
	 */
	async validate(settings: Record<string, any>): Promise<{ valid: boolean; errors?: string[] }> {
		const errors: string[] = []

		for (const required of this.requiredSettings) {
			if (!settings[required]) {
				errors.push(`Missing required setting: ${required}`)
			}
		}

		if (errors.length > 0) {
			return { valid: false, errors }
		}

		return { valid: true }
	}

	/**
	 * Convert data rows to appropriate format for export
	 */
	protected convertData(data: any[]): any[] {
		return data.map((row) => {
			const converted: any = {}
			for (const [key, value] of Object.entries(row)) {
				converted[key] = this.convertValue(value)
			}
			return converted
		})
	}

	/**
	 * Convert individual values to appropriate types
	 */
	protected convertValue(value: any): any {
		if (value === null || value === undefined) {
			return null
		}
		if (value instanceof Date) {
			return value.toISOString()
		}
		if (typeof value === 'object') {
			return JSON.stringify(value)
		}
		return value
	}

	/**
	 * Main export method - must be implemented by subclasses
	 */
	abstract export(data: any[], config: ExportConfig): Promise<ExportResult>

	/**
	 * Log export operation
	 */
	protected logExport(
		config: ExportConfig,
		recordCount: number,
		destination: string,
		status: 'success' | 'failed',
	) {
		const message = `[${this.type}] Exported ${recordCount} records for run ${config.pipelineRunId} to ${destination}`
		if (status === 'success') {
			this.logger.log(message)
		} else {
			this.logger.error(message)
		}
	}
}
