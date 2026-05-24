import type { FastifyInstance } from 'fastify'

export function registerSecurityHook(fastify: FastifyInstance, allowedHosts: string[]) {
	const normalizedAllowedHosts = allowedHosts.filter(Boolean)

	fastify.addHook('onRequest', (request, reply, done) => {
		const host = request.headers.host?.split(':')[0]
		if (normalizedAllowedHosts.length > 0 && host && !normalizedAllowedHosts.includes(host)) {
			reply.code(400).send('Bad Request: Invalid Host Header')
			return
		}
		done()
	})
}
