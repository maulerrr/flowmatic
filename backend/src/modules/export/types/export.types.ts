/**
 * Export adapter types and interfaces
 */

export enum ExportAdapterType {
	POSTGRES = 'postgres',
	MONGODB = 'mongodb',
	HUGGINGFACE = 'huggingface',
	CSV = 'csv',
	JSON = 'json',
	PARQUET = 'parquet',
}

export interface ExportConfig {
	adapterType: ExportAdapterType
	organizationId: string
	pipelineRunId: string
	fileName: string
	settings: Record<string, unknown>
	saveCredentials?: boolean
}

export interface ExportResult {
	success: boolean
	adapterType: ExportAdapterType
	fileName: string
	destination: string
	recordsExported: number
	message: string
	metadata?: Record<string, unknown>
}

export interface ExportAdapter {
	type: ExportAdapterType
	name: string
	description: string
	requiredSettings: string[]

	validate(settings: Record<string, unknown>): Promise<{ valid: boolean; errors?: string[] }>
	export(data: Record<string, unknown>[], config: ExportConfig): Promise<ExportResult>
}

/**
 * PostgreSQL specific types
 */
export interface PostgresConfig {
	host: string
	port: number
	username: string
	password: string
	database: string
	table: string
	ifExists?: 'replace' | 'append' | 'fail' // default: 'append'
}

/**
 * MongoDB specific types
 */
export interface MongoDBConfig {
	uri: string
	database: string
	collection: string
	ifExists?: 'replace' | 'append' | 'fail' // default: 'append'
}

/**
 * Hugging Face specific types
 */
export interface HuggingFaceConfig {
	token: string
	repoName: string
	/** @deprecated Hourly parts are written under data/hourly/ automatically. */
	fileName?: string
	private?: boolean
	commitMessage?: string
	ifExists?: 'replace' | 'append' | 'fail'
}

/**
 * CSV export config
 */
export interface CSVConfig {
	includeIndex?: boolean
	delimiter?: string
}

/**
 * JSON export config
 */
export interface JSONConfig {
	prettyPrint?: boolean
	includeIndex?: boolean
}

/**
 * Parquet export config
 */
export interface ParquetConfig {
	compression?: 'snappy' | 'gzip' | 'brotli' | 'lz4' | 'zstd'
	rowGroupSize?: number
}
