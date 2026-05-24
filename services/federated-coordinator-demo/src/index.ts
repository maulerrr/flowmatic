import { loadConfig } from './config'
import { CoordinatorStore } from './store'

const config = loadConfig()
const store = new CoordinatorStore()

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

function authorize(req: Request, origin: string | null) {
	if (!config.apiKey) return null
	const headerKey = req.headers.get('x-api-key') ?? req.headers.get('authorization')?.replace(/^Bearer\s+/i, '')
	if (headerKey !== config.apiKey) {
		return jsonResponse({ error: 'Unauthorized' }, { status: 401 }, origin)
	}
	return null
}

const server = Bun.serve({
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
					service: 'flowmatic-federated-coordinator-demo',
					globalModelVersion: store.globalModelVersion,
					currentRoundId: store.currentRoundId,
					registrations: store.registrationCount,
				},
				{},
				origin,
			)
		}

		if (url.pathname === '/api/v1/coordinator' && req.method === 'POST') {
			return req
				.json()
				.then(body => jsonResponse(store.handleMessage(body), {}, origin))
				.catch(() => jsonResponse({ error: 'Invalid JSON body' }, { status: 400 }, origin))
		}

		if (url.pathname === '/ws') {
			const upgraded = serverRef.upgrade(req, { data: { nodeId: url.searchParams.get('nodeId') ?? 'demo-node' } })
			if (upgraded) return undefined
			return jsonResponse({ error: 'WebSocket upgrade failed' }, { status: 400 }, origin)
		}

		return jsonResponse({ error: 'Not found' }, { status: 404 }, origin)
	},
	websocket: {
		open(ws) {
			const nodeId = (ws.data as { nodeId: string }).nodeId
			const registration = store.register(nodeId)
			ws.send(JSON.stringify({ type: 'registered', ...registration }))
		},
		message(ws, message) {
			try {
				const body = JSON.parse(String(message))
				const response = store.handleMessage(body)
				ws.send(JSON.stringify(response))
			} catch {
				ws.send(JSON.stringify({ error: 'Invalid JSON message' }))
			}
		},
	},
})

console.log(
	`Federated coordinator demo on http://${server.hostname}:${server.port} (HTTP POST /api/v1/coordinator, WS /ws)`,
)
