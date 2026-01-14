import { Controller, Post, Get, Body, Res, UseGuards, HttpCode, Req } from '@nestjs/common'
import { Response, Request } from 'express'
import { ApiTags } from '@nestjs/swagger'
import { AuthContextService } from './auth-context.service'
import { AuthGuard } from './auth.guard'

@ApiTags('auth')
@Controller('auth')
export class AuthController {
	constructor(private readonly authContext: AuthContextService) {}

	@Post('change-password')
	@UseGuards(AuthGuard)
	async changePassword(
		@Body() body: { password: string },
		@Res() res: Response,
		@Req() req: Request,
	): Promise<void> {
		const { userId } = req.authContext!

		if (!body.password || body.password.length < 6) {
			res.status(400).json({ success: false, error: 'Password must be at least 6 characters' })
			return
		}

		await this.authContext.changePassword(userId, body.password)

		res.json({ success: true, message: 'Password updated successfully' })
	}

	@Post('delete-account')
	@UseGuards(AuthGuard)
	async deleteAccount(@Res() res: Response, @Req() req: Request): Promise<void> {
		const { userId, organizationId } = req.authContext!

		await this.authContext.deleteAccount(userId, organizationId)

		res.clearCookie('flowmatic_session')
		res.json({ success: true, message: 'Account deleted successfully' })
	}

	@Post('login')
	@HttpCode(200)
	async login(
		@Body() body: { email: string; password?: string },
		@Res() res: Response,
	): Promise<void> {
		// For MVP: simple email-based login (SSO provider will validate password)
		// In production: integrate with OAuth2/OIDC provider
		const { user } = await this.authContext.getOrCreateUser(body.email)
		const token = await this.authContext.createSession(user.id)

		res.cookie('flowmatic_session', token, {
			httpOnly: true,
			secure: process.env.NODE_ENV === 'production',
			sameSite: 'lax',
			maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
		})

		res.json({
			success: true,
			user: {
				id: user.id,
				email: user.email,
				displayName: user.displayName,
				organizationId: user.organizationId,
				role: user.role,
			},
		})
	}

	@Get('profile')
	@UseGuards(AuthGuard)
	async getProfile(@Res() res: Response, @Req() req: Request): Promise<void> {
		const { userId } = req.authContext!
		const user = await this.authContext.getUser(userId)

		if (!user) {
			res.status(404).json({ success: false, error: 'User not found' })
			return
		}

		res.json({
			success: true,
			data: user,
		})
	}

	@Post('logout')
	@UseGuards(AuthGuard)
	@HttpCode(200)
	async logout(@Res() res: Response, @Req() req: Request): Promise<void> {
		const cookies = (req as unknown as { cookies: Record<string, string> }).cookies
		const token = cookies?.['flowmatic_session']
		if (token) {
			await this.authContext.invalidateSession(token)
		}

		res.clearCookie('flowmatic_session')
		res.json({ success: true })
	}
}
