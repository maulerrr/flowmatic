import { BaseExportAdapter } from './base.adapter'
import { ExportAdapterType, ExportConfig, ExportResult, CSVConfig } from '../types/export.types'
import { StorageService } from '../../storage/storage.service'
import { rowsToCsv } from 'src/common/utils/csv-parser.util'

/**
 * CSV Export Adapter
 * Exports data as CSV file to S3 storage
 */
export class CSVExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.CSV
	name = 'CSV File'
	description = 'Export data as a CSV file'
	requiredSettings = []

	constructor(private storageService: StorageService) {
		super()
	}

	async export(data: Record<string, unknown>[], config: ExportConfig): Promise<ExportResult> {
		const csvConfig = config.settings as CSVConfig

		try {
			const convertedData = this.convertData(data)

			if (convertedData.length === 0) {
				return {
					success: true,
					adapterType: this.type,
					fileName: config.fileName,
					destination: 'S3 Storage',
					recordsExported: 0,
					message: 'No data to export',
				}
			}

			const csvContent = rowsToCsv(convertedData, { delimiter: csvConfig.delimiter || ',' })
			const s3Key = this.storageService.generateS3Key(
				config.organizationId,
				`${config.fileName}_${Date.now()}`,
				'exports',
			)

			await this.storageService.uploadFileToS3({
				bucket: this.storageService.defaultBucket,
				key: s3Key,
				body: Buffer.from(csvContent),
				contentType: 'text/csv',
				metadata: {
					organizationId: config.organizationId,
					pipelineRunId: config.pipelineRunId,
				},
			})

			this.logExport(config, convertedData.length, s3Key, 'success')

			return {
				success: true,
				adapterType: this.type,
				fileName: config.fileName,
				destination: s3Key,
				recordsExported: convertedData.length,
				message: `Successfully exported ${convertedData.length} records to CSV`,
				metadata: { s3Key },
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, 'CSV file', 'failed')
			throw new Error(`CSV export failed: ${message}`)
		}
	}
}

/**
 * JSON Export Adapter
 * Exports data as JSON file to S3 storage
 */
export class JSONExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.JSON
	name = 'JSON File'
	description = 'Export data as a JSON file'
	requiredSettings = []

	constructor(private storageService: StorageService) {
		super()
	}

	async export(data: Record<string, unknown>[], config: ExportConfig): Promise<ExportResult> {
		const jsonConfig = config.settings

		try {
			const convertedData = this.convertData(data)

			if (convertedData.length === 0) {
				return {
					success: true,
					adapterType: this.type,
					fileName: config.fileName,
					destination: 'S3 Storage',
					recordsExported: 0,
					message: 'No data to export',
				}
			}

			const jsonContent = JSON.stringify(convertedData, null, jsonConfig?.prettyPrint ? 2 : 0)
			const s3Key = this.storageService.generateS3Key(
				config.organizationId,
				`${config.fileName}_${Date.now()}`,
				'exports',
			)

			await this.storageService.uploadFileToS3({
				bucket: this.storageService.defaultBucket,
				key: s3Key,
				body: Buffer.from(jsonContent),
				contentType: 'application/json',
				metadata: {
					organizationId: config.organizationId,
					pipelineRunId: config.pipelineRunId,
				},
			})

			this.logExport(config, convertedData.length, s3Key, 'success')

			return {
				success: true,
				adapterType: this.type,
				fileName: config.fileName,
				destination: s3Key,
				recordsExported: convertedData.length,
				message: `Successfully exported ${convertedData.length} records to JSON`,
				metadata: { s3Key },
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, 'JSON file', 'failed')
			throw new Error(`JSON export failed: ${message}`)
		}
	}
}
