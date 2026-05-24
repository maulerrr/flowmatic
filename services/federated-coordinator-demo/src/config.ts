export interface DemoConfig {
	port: number
	host: string
	apiKey: string | null
	corsOrigins: string[]
}

export function loadConfig(): DemoConfig {
	const port = Number(process.env.PORT ?? '8092')
	const host = process.env.HOST ?? '0.0.0.0'
	const apiKey = process.env.API_KEY?.trim() || null
	const corsOrigins = (process.env.CORS_ORIGINS ?? '*')
		.split(',')
		.map(origin => origin.trim())
		.filter(Boolean)

	return {
		port: Number.isFinite(port) ? port : 8092,
		host,
		apiKey,
		corsOrigins,
	}
}
