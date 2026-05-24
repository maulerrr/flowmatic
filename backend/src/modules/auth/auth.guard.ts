import { CanActivate, ExecutionContext, Injectable, UnauthorizedException } from '@nestjs/common'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { AuthContextService } from './auth-context.service'

@Injectable()
export class AuthGuard implements CanActivate {
	constructor(private readonly authContext: AuthContextService) {}

	async canActivate(context: ExecutionContext): Promise<boolean> {
		const request = context.switchToHttp().getRequest<AuthenticatedRequest>()
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

	private extractTokenFromCookie(request: AuthenticatedRequest): string | null {
		const cookies = request.cookies
		const token = cookies?.['flowmatic_session']
		return typeof token === 'string' ? token : null
	}
}
