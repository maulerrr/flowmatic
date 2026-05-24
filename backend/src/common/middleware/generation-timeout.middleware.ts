import type { FastifyInstance } from 'fastify'

/**
 * Registers a Fastify hook that sets a longer socket timeout for
 * long-running generation endpoints (paths containing `/generate`).
 */
export function registerGenerationTimeoutHook(
	fastify: FastifyInstance,
	timeoutMs: number,
) {
	try {
		if (Number.isNaN(timeoutMs) || timeoutMs <= 0) return

		fastify.addHook('onRequest', (request, reply, done) => {
			const path = request.url || ''
			if (/\/generate(\/|$)/.test(String(path))) {
				if (typeof reply.raw.setTimeout === 'function') {
					reply.raw.setTimeout(timeoutMs)
				}
				console.log(`Set per-request timeout ${timeoutMs}ms for ${path}`)
			}
			done()
		})
	} catch (err) {
		console.warn('Could not create generation timeout middleware', err)
	}
}
