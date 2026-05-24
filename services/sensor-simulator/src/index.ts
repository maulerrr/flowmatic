import { loadConfig } from './config'
import {
	createBatch,
	createPayload,
	isSensorKind,
	SENSOR_KINDS,
	type SensorKind,
} from './payload-generator'

const config = loadConfig()

function corsHeaders(origin: string | null): Record<string, string> {
	const allowed =
		config.corsOrigins.includes('*') ||
		(origin !== null && config.corsOrigins.includes(origin))
	return {
		'Access-Control-Allow-Origin': allowed ? (origin ?? '*') : config.corsOrigins[0] ?? '*',
		'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
	}
}

function jsonResponse(body: unknown, init: ResponseInit = {}, origin: string | null = null) {
	return new Response(JSON.stringify(body), {
		...init,
		headers: {
			'Content-Type': 'application/json',
			...corsHeaders(origin),
			...(init.headers ?? {}),
		},
	})
}

function unauthorized(origin: string | null) {
	return jsonResponse({ error: 'Unauthorized' }, { status: 401 }, origin)
}

function authorize(req: Request, origin: string | null) {
	if (!config.apiKey) return null
	const headerKey = req.headers.get('x-api-key') ?? req.headers.get('authorization')?.replace(/^Bearer\s+/i, '')
	if (headerKey !== config.apiKey) return unauthorized(origin)
	return null
}

function parseSensorKind(url: URL): SensorKind {
	const kind = url.searchParams.get('sensorKind') ?? url.searchParams.get('kind') ?? 'iot'
	return isSensorKind(kind) ? kind : 'iot'
}

function parseLimit(url: URL) {
	const raw = Number(url.searchParams.get('limit') ?? '1')
	return Number.isFinite(raw) ? raw : 1
}

function parseInterval(url: URL) {
	const raw = Number(url.searchParams.get('intervalMs') ?? '1000')
	if (!Number.isFinite(raw)) return 1000
	return Math.min(Math.max(raw, 250), 60000)
}

function buildPresets(baseUrl: string) {
	return SENSOR_KINDS.reduce<Record<string, { httpPollUrl: string; websocketUrl: string }>>((acc, kind) => {
		acc[kind] = {
			httpPollUrl: `${baseUrl}/api/v1/poll?sensorKind=${kind}&limit=1`,
			websocketUrl: `${baseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=${kind}`,
		}
		return acc
	}, {})
}

const server = Bun.serve<{ sensorKind: SensorKind; intervalMs: number; timer: ReturnType<typeof setInterval> | null }>({
	hostname: config.host,
	port: config.port,
	fetch(req, serverRef) {
		const url = new URL(req.url)
		const origin = req.headers.get('origin')

		if (req.method === 'OPTIONS') {
			return new Response(null, { status: 204, headers: corsHeaders(origin) })
		}

		const authError = authorize(req, origin)
		if (authError) return authError

		if (url.pathname === '/health') {
			return jsonResponse(
				{
					status: 'ok',
					service: 'flowmatic-sensor-simulator',
					sensorKinds: SENSOR_KINDS,
					timestamp: new Date().toISOString(),
				},
				{},
				origin,
			)
		}

		if (url.pathname === '/api/v1/presets') {
			const baseUrl = `${url.protocol}//${url.host}`
			return jsonResponse(
				{
					baseUrl,
					defaultLocation: config.defaultLocation,
					sensorKinds: SENSOR_KINDS,
					presets: buildPresets(baseUrl),
				},
				{},
				origin,
			)
		}

		if (url.pathname === '/api/v1/poll' && req.method === 'GET') {
			const sensorKind = parseSensorKind(url)
			const limit = parseLimit(url)
			const events = createBatch(sensorKind, 'HTTP_POLLING', config.defaultLocation, limit)
			return jsonResponse(
				{
					events,
					sensorKind,
					count: events.length,
					generatedAt: new Date().toISOString(),
				},
				{},
				origin,
			)
		}

		if (url.pathname === '/api/v1/emit' && req.method === 'POST') {
			const sensorKind = parseSensorKind(url)
			const event = createPayload(sensorKind, 'HTTP_POLLING', config.defaultLocation)
			return jsonResponse({ event, sensorKind }, {}, origin)
		}

		if (url.pathname === '/ws') {
			const upgraded = serverRef.upgrade(req, {
				data: {
					sensorKind: parseSensorKind(url),
					intervalMs: parseInterval(url),
					timer: null as ReturnType<typeof setInterval> | null,
				},
			})
			if (upgraded) return undefined
			return jsonResponse({ error: 'WebSocket upgrade failed' }, { status: 400 }, origin)
		}

		return jsonResponse({ error: 'Not found' }, { status: 404 }, origin)
	},
	websocket: {
		open(ws) {
			const data = ws.data
			const send = () => {
				const payload = createPayload(data.sensorKind, 'WEBSOCKET', config.defaultLocation)
				ws.send(JSON.stringify(payload))
			}
			send()
			data.timer = setInterval(send, data.intervalMs)
		},
		close(ws) {
			const data = ws.data
			if (data.timer) clearInterval(data.timer)
		},
		message(ws, message) {
			const data = ws.data
			if (String(message).trim().toLowerCase() === 'ping') {
				ws.send(JSON.stringify({ type: 'pong', sensorKind: data.sensorKind, timestamp: new Date().toISOString() }))
			}
		},
	},
})

console.log(
	`Sensor simulator listening on http://${server.hostname}:${server.port} (WS: ws://${server.hostname}:${server.port}/ws)`,
)
