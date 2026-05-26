export interface SimulatorConfig {
	port: number
	host: string
	defaultLocation: string
	apiKey: string | null
	corsOrigins: string[]
	astanaDatasetPath: string
}

export function loadConfig(): SimulatorConfig {
	const port = Number(process.env.PORT ?? '8091')
	const host = process.env.HOST ?? '0.0.0.0'
	const defaultLocation = process.env.DEFAULT_LOCATION ?? 'Astana'
	const apiKey = process.env.API_KEY?.trim() || null
	const corsOrigins = (process.env.CORS_ORIGINS ?? '*')
		.split(',')
		.map(origin => origin.trim())
		.filter(Boolean)
	const astanaDatasetPath =
		process.env.ASTANA_DATASET_PATH?.trim() || '/data/astana_synthetic_data.csv'

	return {
		port: Number.isFinite(port) ? port : 8091,
		host,
		defaultLocation,
		apiKey,
		corsOrigins,
		astanaDatasetPath,
	}
}
