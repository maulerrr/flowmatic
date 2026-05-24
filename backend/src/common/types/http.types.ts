import { FastifyRequest } from 'fastify'
import { AuthContext } from 'src/modules/auth/auth-context.service'

export type AuthenticatedRequest = FastifyRequest & {
	authContext?: AuthContext
	cookies?: Record<string, string>
}
