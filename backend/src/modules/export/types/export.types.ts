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
	settings: Record<string, any>
}

export interface ExportResult {
	success: boolean
	adapterType: ExportAdapterType
	fileName: string
	destination: string
	recordsExported: number
	message: string
	metadata?: Record<string, any>
}

export interface ExportAdapter {
	type: ExportAdapterType
	name: string
	description: string
	requiredSettings: string[]

	validate(settings: Record<string, any>): Promise<{ valid: boolean; errors?: string[] }>
	export(data: any[], config: ExportConfig): Promise<ExportResult>
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
	fileName: string
	private?: boolean
	commitMessage?: string
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
