import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common'
import { Request } from 'express'
import { AuthContextService, AuthContext } from './auth-context.service'

declare module 'express' {
	interface Request {
		authContext?: AuthContext
	}
}

@Injectable()
export class AuthGuard implements CanActivate {
	constructor(private readonly authContext: AuthContextService) {}

	async canActivate(context: ExecutionContext): Promise<boolean> {
		const request = context.switchToHttp().getRequest<Request>()
		const token = this.extractTokenFromCookie(request)

		if (!token) {
			throw new UnauthorizedException('No authentication token found')
		}

		const authContext = await this.authContext.validateSession(token)

		if (!authContext) {
			throw new UnauthorizedException('Invalid or expired session')
		}

		request.authContext = authContext
		return true
	}

	private extractTokenFromCookie(request: Request): string | null {
		const cookies = (request as unknown as { cookies?: Record<string, unknown> }).cookies
		const token = cookies?.['flowmatic_session']
		return typeof token === 'string' ? token : null
	}
}
