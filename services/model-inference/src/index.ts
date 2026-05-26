const port = Number(process.env.PORT ?? 8093)

function json(body: unknown, init: ResponseInit = {}) {
	return new Response(JSON.stringify(body), {
		...init,
		headers: {
			'Content-Type': 'application/json',
			...(init.headers ?? {}),
		},
	})
}

function normalizeInputs(inputs: unknown) {
	if (typeof inputs === 'string') return inputs
	if (inputs && typeof inputs === 'object') return JSON.stringify(inputs)
	return String(inputs ?? '')
}

async function inferWithHuggingFace(modelId: string, token: string, inputs: unknown) {
	const response = await fetch(`https://api-inference.huggingface.co/models/${modelId}`, {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${token}`,
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({ inputs: normalizeInputs(inputs) }),
	})

	const text = await response.text()
	let parsed: unknown = text
	try {
		parsed = text ? JSON.parse(text) : null
	} catch {
		parsed = text
	}

	if (!response.ok) {
		const message =
			typeof parsed === 'object' &&
			parsed !== null &&
			'error' in parsed &&
			typeof (parsed as { error?: unknown }).error === 'string'
				? (parsed as { error: string }).error
				: `Hugging Face inference failed (${response.status})`
		throw new Error(message)
	}

	return {
		provider: 'huggingface-serverless',
		modelId,
		latencyMs: null,
		result: parsed,
	}
}

const server = Bun.serve({
	port,
	async fetch(req) {
		const url = new URL(req.url)
		if (req.method === 'GET' && url.pathname === '/health') {
			return json({ status: 'ok', service: 'model-inference', port })
		}

		if (req.method === 'POST' && url.pathname === '/v1/infer') {
			const body = (await req.json().catch(() => null)) as
				| { modelId?: string; token?: string; inputs?: unknown }
				| null
			if (!body?.modelId?.trim() || !body.token?.trim()) {
				return json({ error: 'modelId and token are required' }, { status: 400 })
			}

			const started = performance.now()
			try {
				const output = await inferWithHuggingFace(body.modelId.trim(), body.token.trim(), body.inputs)
				return json({
					...output,
					latencyMs: Math.round(performance.now() - started),
				})
			} catch (error) {
				return json(
					{ error: error instanceof Error ? error.message : 'Inference failed' },
					{ status: 502 },
				)
			}
		}

		return json({ error: 'Not found' }, { status: 404 })
	},
})

console.log(`Model inference service listening on http://localhost:${server.port}`)
