import {
	Controller,
	Post,
	Get,
	Body,
	Param,
	Patch,
	Delete,
	Res,
	UseGuards,
	HttpCode,
	Req,
	NotFoundException,
	Inject,
	forwardRef,
} from '@nestjs/common'
import { FastifyReply } from 'fastify'
import { ApiTags } from '@nestjs/swagger'
import { AuthContextService } from './auth-context.service'
import { AuthGuard } from './auth.guard'
import { AppConfigService } from 'src/common/config/config.service'
import { ChangePasswordDto } from './dto/change-password.dto'
import { LoginDto } from './dto/login.dto'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { RegisterDto } from './dto/register.dto'
import { UpdateProfileDto } from './dto/update-profile.dto'
import { CreateOrganizationDto } from './dto/create-organization.dto'
import { UpdateOrganizationDto } from './dto/update-organization.dto'
import { InviteMemberDto } from './dto/invite-member.dto'
import { SwitchOrganizationDto } from './dto/switch-organization.dto'
import { DeleteOrganizationDto } from './dto/delete-organization.dto'
import { ExportService } from '../export/export.service'
import { whoAmI } from '@huggingface/hub'

@ApiTags('auth')
@Controller('auth')
export class AuthController {
	constructor(
		private readonly authContext: AuthContextService,
		private readonly config: AppConfigService,
		@Inject(forwardRef(() => ExportService))
		private readonly exportService: ExportService,
	) {}

	@Post('change-password')
	@UseGuards(AuthGuard)
	async changePassword(@Body() body: ChangePasswordDto, @Req() req: AuthenticatedRequest) {
		const { userId } = req.authContext!

		await this.authContext.changePassword(userId, body.password)

		return { success: true, message: 'Password updated successfully' }
	}

	@Post('delete-account')
	@UseGuards(AuthGuard)
	async deleteAccount(
		@Res({ passthrough: true }) res: FastifyReply,
		@Req() req: AuthenticatedRequest,
	) {
		const { userId, organizationId } = req.authContext!

		await this.authContext.deleteAccount(userId, organizationId)

		this.clearSessionCookies(res)
		return { success: true, message: 'Account deleted successfully' }
	}

	@Post('login')
	@HttpCode(200)
	async login(@Body() body: LoginDto, @Res({ passthrough: true }) res: FastifyReply) {
		const user = await this.authContext.login(body.email, body.password)
		const token = await this.authContext.createSession(user.id)

		this.clearSessionCookies(res)
		res.setCookie('flowmatic_session', token, {
			httpOnly: true,
			secure: this.config.isProduction,
			sameSite: 'lax',
			path: '/',
			maxAge: 30 * 24 * 60 * 60,
		})

		return {
			success: true,
			data: {
				user: {
					id: user.id,
					email: user.email,
					displayName: user.displayName,
					organizationId: user.organizationId,
					role: user.role,
				},
			},
		}
	}

	@Post('register')
	@HttpCode(201)
	async register(@Body() body: RegisterDto, @Res({ passthrough: true }) res: FastifyReply) {
		const user = await this.authContext.register(body)
		if (body.huggingFaceToken?.trim()) {
			await this.exportService.saveOrganizationHuggingFaceToken(
				user.organizationId,
				body.huggingFaceToken.trim(),
			)
		}
		const token = await this.authContext.createSession(user.id)

		this.clearSessionCookies(res)
		res.setCookie('flowmatic_session', token, {
			httpOnly: true,
			secure: this.config.isProduction,
			sameSite: 'lax',
			path: '/',
			maxAge: 30 * 24 * 60 * 60,
		})

		return {
			success: true,
			data: {
				user: {
					id: user.id,
					email: user.email,
					displayName: user.displayName,
					organizationId: user.organizationId,
					role: user.role,
				},
			},
		}
	}

	@Post('switch-organization')
	@UseGuards(AuthGuard)
	async switchOrganization(@Body() body: SwitchOrganizationDto, @Req() req: AuthenticatedRequest) {
		const organization = await this.authContext.switchOrganization(
			req.authContext!.userId,
			body.organizationId,
		)
		return { success: true, data: organization }
	}

	@Get('organizations')
	@UseGuards(AuthGuard)
	async listOrganizations(@Req() req: AuthenticatedRequest) {
		const organizations = await this.authContext.listOrganizations(req.authContext!.userId)
		return { success: true, data: organizations }
	}

	@Post('organizations')
	@UseGuards(AuthGuard)
	async createOrganization(@Body() body: CreateOrganizationDto, @Req() req: AuthenticatedRequest) {
		const organization = await this.authContext.createOrganization(
			req.authContext!.userId,
			body.name,
		)
		return { success: true, data: organization }
	}

	@Patch('organizations/:organizationId')
	@UseGuards(AuthGuard)
	async updateOrganization(
		@Param('organizationId') organizationId: string,
		@Body() body: UpdateOrganizationDto,
		@Req() req: AuthenticatedRequest,
	) {
		const organization = await this.authContext.updateOrganization(
			req.authContext!.userId,
			organizationId,
			body.name,
		)
		return { success: true, data: organization }
	}

	@Delete('organizations/:organizationId')
	@UseGuards(AuthGuard)
	async deleteOrganization(
		@Param('organizationId') organizationId: string,
		@Body() body: DeleteOrganizationDto,
		@Req() req: AuthenticatedRequest,
	) {
		const result = await this.authContext.deleteOrganization(
			req.authContext!.userId,
			organizationId,
			body.confirmationName,
		)
		return { success: true, data: result }
	}

	@Get('organizations/:organizationId/members')
	@UseGuards(AuthGuard)
	async listMembers(
		@Param('organizationId') organizationId: string,
		@Req() req: AuthenticatedRequest,
	) {
		const members = await this.authContext.listMembers(req.authContext!.userId, organizationId)
		return { success: true, data: members }
	}

	@Post('organizations/:organizationId/invitations')
	@UseGuards(AuthGuard)
	async inviteMember(
		@Param('organizationId') organizationId: string,
		@Body() body: InviteMemberDto,
		@Req() req: AuthenticatedRequest,
	) {
		const invitation = await this.authContext.inviteMember(
			req.authContext!.userId,
			organizationId,
			body.email,
			body.role,
		)
		return { success: true, data: invitation }
	}

	@Get('invitations')
	@UseGuards(AuthGuard)
	async listInvitations(@Req() req: AuthenticatedRequest) {
		const invitations = await this.authContext.listMyInvitations(req.authContext!.userId)
		return { success: true, data: invitations }
	}

	@Post('invitations/:invitationId/accept')
	@UseGuards(AuthGuard)
	async acceptInvitation(
		@Param('invitationId') invitationId: string,
		@Req() req: AuthenticatedRequest,
	) {
		const organization = await this.authContext.acceptInvitation(
			req.authContext!.userId,
			invitationId,
		)
		return { success: true, data: organization }
	}

	@Post('invitations/:invitationId/decline')
	@UseGuards(AuthGuard)
	async declineInvitation(
		@Param('invitationId') invitationId: string,
		@Req() req: AuthenticatedRequest,
	) {
		await this.authContext.declineInvitation(req.authContext!.userId, invitationId)
		return { success: true, data: { declined: true } }
	}

	@Patch('profile')
	@UseGuards(AuthGuard)
	async updateProfile(@Body() body: UpdateProfileDto, @Req() req: AuthenticatedRequest) {
		const user = await this.authContext.updateProfile(req.authContext!.userId, body)
		return {
			success: true,
			data: {
				id: user.id,
				email: user.email,
				displayName: user.displayName,
				organizationId: user.organizationId,
				role: user.role,
				createdAt: user.createdAt,
				updatedAt: user.updatedAt,
			},
		}
	}

	@Get('profile')
	@UseGuards(AuthGuard)
	async getProfile(@Req() req: AuthenticatedRequest) {
		const { userId } = req.authContext!
		const user = await this.authContext.getUser(userId)

		if (!user) {
			throw new NotFoundException('User not found')
		}

		return {
			success: true,
			data: {
				...user,
				huggingFaceIntegration: await this.getHuggingFaceIntegrationStatus(
					req.authContext!.organizationId,
				),
			},
		}
	}

	@Post('logout')
	@HttpCode(200)
	async logout(@Res({ passthrough: true }) res: FastifyReply, @Req() req: AuthenticatedRequest) {
		const cookies = req.cookies
		const token = cookies?.['flowmatic_session']
		if (token) {
			await this.authContext.invalidateSession(token)
		}

		this.clearSessionCookies(res)
		return { success: true, data: { loggedOut: true } }
	}

	private clearSessionCookies(res: FastifyReply) {
		for (const path of ['/', '/api/v1', '/api/v1/auth', '/auth']) {
			res.clearCookie('flowmatic_session', { path })
		}
	}

	private async getHuggingFaceIntegrationStatus(organizationId: string) {
		const token = await this.exportService.getOrganizationHuggingFaceToken(organizationId)
		if (!token) return { configured: false }
		const tokenPreview =
			await this.exportService.getOrganizationHuggingFaceTokenPreview(organizationId)
		try {
			const profile = await whoAmI({ accessToken: token })
			return { configured: true, username: profile.name, tokenPreview }
		} catch {
			return { configured: true, tokenPreview }
		}
	}
}
