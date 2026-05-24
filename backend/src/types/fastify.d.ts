import { AuthContext } from '../modules/auth/auth-context.service'

declare module 'fastify' {
	interface FastifyRequest {
		authContext?: AuthContext
	}
}
