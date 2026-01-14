import { BaseExportAdapter } from './base.adapter'
import { ExportAdapterType, ExportConfig, ExportResult, PostgresConfig } from '../types/export.types'

/**
 * PostgreSQL Export Adapter
 * Uploads cleaned data to a PostgreSQL database
 */
export class PostgresExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.POSTGRES
	name = 'PostgreSQL Database'
	description = 'Export data to a PostgreSQL database table'
	requiredSettings = ['host', 'port', 'username', 'password', 'database', 'table']

	async validate(settings: Record<string, any>): Promise<{ valid: boolean; errors?: string[] }> {
		const baseValidation = await super.validate(settings)
		if (!baseValidation.valid) return baseValidation

		const errors: string[] = []
		const config = settings as PostgresConfig

		if (!Number.isInteger(config.port) || config.port < 1 || config.port > 65535) {
			errors.push('Port must be a valid port number (1-65535)')
		}

		if (!config.host || config.host.trim().length === 0) {
			errors.push('Host cannot be empty')
		}

		if (!config.database || config.database.trim().length === 0) {
			errors.push('Database name cannot be empty')
		}

		if (!config.table || config.table.trim().length === 0) {
			errors.push('Table name cannot be empty')
		}

		if (errors.length > 0) {
			return { valid: false, errors }
		}

		// Test connection (optional - could be added)
		try {
			// Connection test would go here
		} catch (error) {
			return { valid: false, errors: ['Failed to connect to PostgreSQL database'] }
		}

		return { valid: true }
	}

	/**
	 * Infer PostgreSQL column types from data values
	 */
	private inferColumnTypes(data: any[]): Record<string, string> {
		const typeMap: Record<string, string> = {}

		if (data.length === 0) {
			return typeMap
		}

		const columns = Object.keys(data[0])

		for (const col of columns) {
			const values = data.map((row) => row[col]).filter((v) => v !== null && v !== undefined)

			if (values.length === 0) {
				typeMap[col] = 'TEXT'
				continue
			}

			// Check for boolean
			if (values.every((v) => typeof v === 'boolean')) {
				typeMap[col] = 'BOOLEAN'
				continue
			}

			// Check for integer
			if (values.every((v) => typeof v === 'number' && Number.isInteger(v))) {
				typeMap[col] = 'INTEGER'
				continue
			}

			// Check for float/number
			if (values.every((v) => typeof v === 'number')) {
				typeMap[col] = 'NUMERIC(15,6)'
				continue
			}

			// Check for date/timestamp
			if (values.every((v) => {
				const date = new Date(v)
				return !isNaN(date.getTime())
			})) {
				typeMap[col] = 'TIMESTAMP'
				continue
			}

			// Default to TEXT
			typeMap[col] = 'TEXT'
		}

		return typeMap
	}

	/**
	 * Create table if it doesn't exist with proper schema
	 */
	private async createTableIfNotExists(
		client: any,
		tableName: string,
		columnTypes: Record<string, string>,
		recreate: boolean = false,
	): Promise<void> {
		const columns = Object.entries(columnTypes)
			.map(([name, type]) => `"${name}" ${type}`)
			.join(',\n    ')

		// If recreate=true, drop the table first to ensure schema matches the incoming CSV exactly
		if (recreate) {
			await client.query(`DROP TABLE IF EXISTS "${tableName}"`)
		}

		const createTableQuery = `
      CREATE TABLE IF NOT EXISTS "${tableName}" (
        ${columns}
      )
    `

		await client.query(createTableQuery)
	}

	async export(data: any[], config: ExportConfig): Promise<ExportResult> {
		const pgConfig = config.settings as PostgresConfig

		try {
			// Dynamic import to avoid hard dependency
			const pgModule = await import('pg')
			const { Pool } = pgModule

			const pool = new Pool({
				host: pgConfig.host,
				port: pgConfig.port,
				user: pgConfig.username,
				password: pgConfig.password,
				database: pgConfig.database,
			})

			const client = await pool.connect()
			try {
				// Use raw data rows to preserve column names and values exactly as in the CSV
				const convertedData = data

				if (convertedData.length === 0) {
					await client.release()
					await pool.end()
					return {
						success: true,
						adapterType: this.type,
						fileName: config.fileName,
						destination: `${pgConfig.host}/${pgConfig.database}.${pgConfig.table}`,
						recordsExported: 0,
						message: 'No data to export',
					}
				}

				// Infer column types from data
				const columnTypes = this.inferColumnTypes(convertedData)

				// Handle table creation and ifExists behavior
				const ifExists = pgConfig.ifExists || 'append'
				const recreate = ifExists === 'replace'

				// Create table; if replace, drop then recreate to match CSV columns exactly
				await this.createTableIfNotExists(client, pgConfig.table, columnTypes, recreate)

				// If append, ensure table exists but keep existing data; if replace, table is already recreated empty

				// Insert data in batches
				const batchSize = 1000
				let inserted = 0

				for (let i = 0; i < convertedData.length; i += batchSize) {
					const batch = convertedData.slice(i, i + batchSize)
					const columns = Object.keys(batch[0])
					const placeholders = batch
						.map(
							(_, idx) =>
								`(${columns
									.map((_, colIdx) => `$${idx * columns.length + colIdx + 1}`)
									.join(',')})`,
						)
						.join(',')
					const values = batch.flatMap((row) => columns.map((col) => row[col]))

					const query = `
            INSERT INTO "${pgConfig.table}" (${columns.map((c) => `"${c}"`).join(',')})
            VALUES ${placeholders}
          `

					await client.query(query, values)
					inserted += batch.length
				}

				this.logExport(config, inserted, `${pgConfig.host}/${pgConfig.database}.${pgConfig.table}`, 'success')

				return {
					success: true,
					adapterType: this.type,
					fileName: config.fileName,
					destination: `${pgConfig.host}/${pgConfig.database}.${pgConfig.table}`,
					recordsExported: inserted,
					message: `Successfully exported ${inserted} records to PostgreSQL`,
					metadata: {
						columnTypes,
						tableName: pgConfig.table,
					},
				}
			} finally {
				await client.release()
				await pool.end()
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, pgConfig.table, 'failed')
			throw new Error(`PostgreSQL export failed: ${message}`)
		}
	}
}
