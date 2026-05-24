import { BaseExportAdapter } from './base.adapter'
import { ExportAdapterType, ExportConfig, ExportResult, MongoDBConfig } from '../types/export.types'

/**
 * MongoDB Export Adapter
 * Uploads cleaned data to a MongoDB collection
 */
export class MongoDBExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.MONGODB
	name = 'MongoDB Database'
	description = 'Export data to a MongoDB collection'
	requiredSettings = ['uri', 'database', 'collection']

	async validate(
		settings: Record<string, unknown>,
	): Promise<{ valid: boolean; errors?: string[] }> {
		const baseValidation = await super.validate(settings)
		if (!baseValidation.valid) return baseValidation

		const errors: string[] = []
		const config = settings as unknown as MongoDBConfig

		if (!config.uri.startsWith('mongodb')) {
			errors.push('URI must be a valid MongoDB connection string')
		}

		if (!config.database || config.database.trim().length === 0) {
			errors.push('Database name cannot be empty')
		}

		if (!config.collection || config.collection.trim().length === 0) {
			errors.push('Collection name cannot be empty')
		}

		if (errors.length > 0) {
			return { valid: false, errors }
		}

		return { valid: true }
	}

	async export(data: Record<string, unknown>[], config: ExportConfig): Promise<ExportResult> {
		const mongoConfig = config.settings as unknown as MongoDBConfig

		try {
			// Dynamic import to avoid hard dependency
			const { MongoClient } = await import('mongodb')

			const client = new MongoClient(mongoConfig.uri) as unknown as {
				connect(): Promise<void>
				close(): Promise<void>
				db(name: string): {
					collection(name: string): {
						deleteMany(filter: unknown): Promise<unknown>
						insertMany(docs: unknown[]): Promise<{ insertedCount: number }>
					}
				}
			}
			await client.connect()

			try {
				const db = client.db(mongoConfig.database)
				const collection = db.collection(mongoConfig.collection)

				const convertedData = this.convertData(data)

				if (convertedData.length === 0) {
					return {
						success: true,
						adapterType: this.type,
						fileName: config.fileName,
						destination: `${mongoConfig.database}.${mongoConfig.collection}`,
						recordsExported: 0,
						message: 'No data to export',
					}
				}

				// Handle collection clearing based on ifExists
				const ifExists = mongoConfig.ifExists || 'append'
				if (ifExists === 'replace') {
					await collection.deleteMany({})
				}

				// Insert documents
				const result = await collection.insertMany(
					convertedData.map(doc => ({
						...doc,
						_importedAt: new Date(),
						_pipelineRunId: config.pipelineRunId,
					})),
				)

				this.logExport(
					config,
					result.insertedCount,
					`${mongoConfig.database}.${mongoConfig.collection}`,
					'success',
				)

				return {
					success: true,
					adapterType: this.type,
					fileName: config.fileName,
					destination: `${mongoConfig.database}.${mongoConfig.collection}`,
					recordsExported: result.insertedCount,
					message: `Successfully exported ${result.insertedCount} records to MongoDB`,
				}
			} finally {
				await client.close()
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, mongoConfig.collection, 'failed')
			throw new Error(`MongoDB export failed: ${message}`)
		}
	}
}
