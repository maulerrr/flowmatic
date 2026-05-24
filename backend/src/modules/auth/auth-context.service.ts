import {
	BadRequestException,
	ForbiddenException,
	Injectable,
	NotFoundException,
	UnauthorizedException,
} from '@nestjs/common'
import { Prisma, User } from '@prisma/client'
import * as bcrypt from 'bcrypt'
import { randomBytes } from 'crypto'
import { PrismaService } from 'src/prisma/prisma.service'

export type UserRole = 'admin' | 'member' | 'viewer'

export interface AuthContext {
	userId: string
	organizationId: string
	role: UserRole
}

@Injectable()
export class AuthContextService {
	private readonly maxOrganizationsPerUser = 3

	constructor(private readonly prisma: PrismaService) {}

	async validateSession(token: string): Promise<AuthContext | null> {
		const session = await this.prisma.session.findUnique({
			where: { token },
		})

		if (!session || session.expiresAt < new Date()) {
			return null
		}

		const user = await this.prisma.user.findUnique({
			where: { id: session.userId },
			include: { memberships: true },
		})

		if (!user) return null

		const activeMembership =
			user.memberships.find(member => member.organizationId === user.organizationId) ??
			user.memberships[0]

		if (!activeMembership) return null

		if (activeMembership.organizationId !== user.organizationId) {
			await this.prisma.user.update({
				where: { id: user.id },
				data: {
					organizationId: activeMembership.organizationId,
					role: activeMembership.role,
				},
			})
		}

		return {
			userId: user.id,
			organizationId: activeMembership.organizationId,
			role: activeMembership.role as UserRole,
		}
	}

	async createSession(userId: string, expiryDays: number = 30): Promise<string> {
		const token = randomBytes(48).toString('hex')

		await this.prisma.session.create({
			data: {
				userId,
				token,
				expiresAt: new Date(Date.now() + expiryDays * 24 * 60 * 60 * 1000),
			},
		})

		return token
	}

	async invalidateSession(token: string): Promise<void> {
		await this.prisma.session.delete({ where: { token } }).catch(() => {})
	}

	async register(input: {
		email: string
		password: string
		displayName: string
		organizationName?: string
	}): Promise<User> {
		const email = input.email.toLowerCase().trim()
		const existing = await this.prisma.user.findUnique({ where: { email } })
		if (existing) {
			throw new BadRequestException('An account with this email already exists')
		}

		const displayName = input.displayName.trim()
		const organizationName =
			input.organizationName?.trim() || `${displayName || email.split('@')[0]}'s Organization`
		const password = await bcrypt.hash(input.password, 10)

		return this.prisma.$transaction(async tx => {
			const organization = await tx.organization.create({
				data: {
					name: organizationName,
					slug: await this.createUniqueSlug(organizationName, tx),
				},
			})

			const user = await tx.user.create({
				data: {
					email,
					password,
					displayName,
					organizationId: organization.id,
					role: 'admin',
				},
			})

			await tx.organizationMember.create({
				data: {
					userId: user.id,
					organizationId: organization.id,
					role: 'admin',
				},
			})

			return user
		})
	}

	async login(email: string, password: string): Promise<User> {
		const user = await this.prisma.user.findUnique({
			where: { email: email.toLowerCase().trim() },
		})

		if (!user) {
			throw new UnauthorizedException('Invalid email or password')
		}

		const passwordMatches = user.password ? await bcrypt.compare(password, user.password) : false
		if (!passwordMatches) {
			throw new UnauthorizedException('Invalid email or password')
		}

		await this.ensureDefaultMembership(user)
		return user
	}

	async getUser(userId: string) {
		const user = await this.prisma.user.findUnique({
			where: { id: userId },
			select: {
				id: true,
				email: true,
				displayName: true,
				organizationId: true,
				role: true,
				createdAt: true,
				updatedAt: true,
				memberships: {
					include: { organization: true },
					orderBy: { createdAt: 'asc' },
				},
			},
		})

		if (!user) return null

		return {
			id: user.id,
			email: user.email,
			displayName: user.displayName,
			organizationId: user.organizationId,
			role: user.role,
			createdAt: user.createdAt,
			updatedAt: user.updatedAt,
			organizations: user.memberships.map(member => ({
				id: member.organization.id,
				name: member.organization.name,
				slug: member.organization.slug,
				role: member.role,
				isActive: member.organizationId === user.organizationId,
			})),
		}
	}

	async updateProfile(userId: string, input: { displayName?: string }) {
		return this.prisma.user.update({
			where: { id: userId },
			data: {
				displayName: input.displayName?.trim(),
			},
			select: {
				id: true,
				email: true,
				displayName: true,
				organizationId: true,
				role: true,
				createdAt: true,
				updatedAt: true,
			},
		})
	}

	async changePassword(userId: string, newPassword: string): Promise<void> {
		const hashedPassword = await bcrypt.hash(newPassword, 10)
		await this.prisma.user.update({
			where: { id: userId },
			data: { password: hashedPassword },
		})
	}

	async listOrganizations(userId: string) {
		await this.ensureUserMemberships(userId)
		const memberships = await this.prisma.organizationMember.findMany({
			where: { userId },
			include: { organization: true },
			orderBy: { createdAt: 'asc' },
		})
		const user = await this.prisma.user.findUniqueOrThrow({ where: { id: userId } })

		return memberships.map(member => ({
			id: member.organization.id,
			name: member.organization.name,
			slug: member.organization.slug,
			role: member.role,
			isActive: member.organizationId === user.organizationId,
			createdAt: member.organization.createdAt,
		}))
	}

	async createOrganization(userId: string, name: string) {
		const membershipCount = await this.prisma.organizationMember.count({ where: { userId } })
		if (membershipCount >= this.maxOrganizationsPerUser) {
			throw new BadRequestException('A user can belong to at most 3 organizations')
		}

		return this.prisma.$transaction(async tx => {
			const organization = await tx.organization.create({
				data: {
					name: name.trim(),
					slug: await this.createUniqueSlug(name, tx),
				},
			})

			await tx.organizationMember.create({
				data: {
					userId,
					organizationId: organization.id,
					role: 'admin',
				},
			})

			await tx.user.update({
				where: { id: userId },
				data: {
					organizationId: organization.id,
					role: 'admin',
				},
			})

			return organization
		})
	}

	async switchOrganization(userId: string, organizationId: string) {
		const membership = await this.prisma.organizationMember.findUnique({
			where: { userId_organizationId: { userId, organizationId } },
			include: { organization: true },
		})

		if (!membership) {
			throw new ForbiddenException('You are not a member of this organization')
		}

		await this.prisma.user.update({
			where: { id: userId },
			data: {
				organizationId,
				role: membership.role,
			},
		})

		return {
			id: membership.organization.id,
			name: membership.organization.name,
			slug: membership.organization.slug,
			role: membership.role,
			isActive: true,
		}
	}

	async updateOrganization(userId: string, organizationId: string, name: string) {
		await this.requireOrganizationAdmin(userId, organizationId)
		return this.prisma.organization.update({
			where: { id: organizationId },
			data: {
				name: name.trim(),
				slug: await this.createUniqueSlug(name),
			},
		})
	}

	async deleteOrganization(userId: string, organizationId: string, confirmationName: string) {
		const membership = await this.requireOrganizationAdmin(userId, organizationId)
		const organization = await this.prisma.organization.findUnique({
			where: { id: organizationId },
		})

		if (!organization) throw new NotFoundException('Organization not found')
		if (confirmationName !== organization.name) {
			throw new BadRequestException('Organization name confirmation does not match')
		}

		const remainingMembership = await this.prisma.organizationMember.findFirst({
			where: {
				userId,
				organizationId: { not: organizationId },
			},
			orderBy: { createdAt: 'asc' },
		})

		if (!remainingMembership) {
			throw new BadRequestException('You cannot delete your last organization')
		}

		await this.prisma.organization.delete({ where: { id: organizationId } })

		await this.prisma.user.update({
			where: { id: userId },
			data: {
				organizationId: remainingMembership.organizationId,
				role: remainingMembership.role,
			},
		})
		return { deleted: true, nextOrganizationId: remainingMembership.organizationId }
	}

	async listMembers(userId: string, organizationId: string) {
		await this.requireOrganizationMember(userId, organizationId)
		return this.prisma.organizationMember.findMany({
			where: { organizationId },
			include: {
				user: {
					select: {
						id: true,
						email: true,
						displayName: true,
						createdAt: true,
					},
				},
			},
			orderBy: { createdAt: 'asc' },
		})
	}

	async inviteMember(
		userId: string,
		organizationId: string,
		email: string,
		role: UserRole = 'member',
	) {
		await this.requireOrganizationAdmin(userId, organizationId)
		const normalizedEmail = email.toLowerCase().trim()
		const invitedUser = await this.prisma.user.findUnique({ where: { email: normalizedEmail } })

		if (invitedUser) {
			const membershipCount = await this.prisma.organizationMember.count({
				where: { userId: invitedUser.id },
			})
			if (membershipCount >= this.maxOrganizationsPerUser) {
				throw new BadRequestException('That user already belongs to 3 organizations')
			}

			const existingMember = await this.prisma.organizationMember.findUnique({
				where: { userId_organizationId: { userId: invitedUser.id, organizationId } },
			})
			if (existingMember) {
				throw new BadRequestException('That user is already a member of this organization')
			}
		}

		return this.prisma.organizationInvitation.create({
			data: {
				email: normalizedEmail,
				role,
				organizationId,
				invitedById: userId,
				token: randomBytes(32).toString('hex'),
				expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
			},
			include: { organization: true },
		})
	}

	async listMyInvitations(userId: string) {
		const user = await this.prisma.user.findUniqueOrThrow({ where: { id: userId } })
		return this.prisma.organizationInvitation.findMany({
			where: {
				email: user.email,
				acceptedAt: null,
				declinedAt: null,
				expiresAt: { gt: new Date() },
			},
			include: {
				organization: true,
				invitedBy: { select: { id: true, email: true, displayName: true } },
			},
			orderBy: { createdAt: 'desc' },
		})
	}

	async acceptInvitation(userId: string, invitationId: string) {
		const user = await this.prisma.user.findUniqueOrThrow({ where: { id: userId } })
		const invitation = await this.prisma.organizationInvitation.findFirst({
			where: {
				id: invitationId,
				email: user.email,
				acceptedAt: null,
				declinedAt: null,
				expiresAt: { gt: new Date() },
			},
			include: { organization: true },
		})

		if (!invitation) {
			throw new NotFoundException('Invitation not found')
		}

		const membershipCount = await this.prisma.organizationMember.count({ where: { userId } })
		const existingMembership = await this.prisma.organizationMember.findUnique({
			where: {
				userId_organizationId: {
					userId,
					organizationId: invitation.organizationId,
				},
			},
		})

		if (!existingMembership && membershipCount >= this.maxOrganizationsPerUser) {
			throw new BadRequestException('A user can belong to at most 3 organizations')
		}

		return this.prisma.$transaction(async tx => {
			if (!existingMembership) {
				await tx.organizationMember.create({
					data: {
						userId,
						organizationId: invitation.organizationId,
						role: invitation.role,
					},
				})
			}
			await tx.organizationInvitation.updateMany({
				where: {
					email: user.email,
					organizationId: invitation.organizationId,
					acceptedAt: null,
					declinedAt: null,
				},
				data: { acceptedAt: new Date() },
			})
			await tx.user.update({
				where: { id: userId },
				data: {
					organizationId: invitation.organizationId,
					role: invitation.role,
				},
			})
			return invitation.organization
		})
	}

	async declineInvitation(userId: string, invitationId: string) {
		const user = await this.prisma.user.findUniqueOrThrow({ where: { id: userId } })
		const invitation = await this.prisma.organizationInvitation.findFirst({
			where: {
				id: invitationId,
				email: user.email,
				acceptedAt: null,
				declinedAt: null,
			},
		})

		if (!invitation) throw new NotFoundException('Invitation not found')

		await this.prisma.organizationInvitation.update({
			where: { id: invitation.id },
			data: { declinedAt: new Date() },
		})
	}

	async deleteAccount(userId: string, organizationId: string): Promise<void> {
		const membership = await this.prisma.organizationMember.findUnique({
			where: { userId_organizationId: { userId, organizationId } },
		})
		if (!membership) return

		const memberCount = await this.prisma.organizationMember.count({ where: { organizationId } })

		if (memberCount === 1) {
			await this.prisma.organization.delete({ where: { id: organizationId } })
			await this.prisma.user.delete({ where: { id: userId } }).catch(() => {})
			return
		}

		await this.prisma.$transaction(async tx => {
			await tx.session.deleteMany({ where: { userId } })
			await tx.organizationMember.delete({
				where: { userId_organizationId: { userId, organizationId } },
			})
			const nextMembership = await tx.organizationMember.findFirst({
				where: { userId },
				orderBy: { createdAt: 'asc' },
			})
			if (nextMembership) {
				await tx.user.update({
					where: { id: userId },
					data: {
						organizationId: nextMembership.organizationId,
						role: nextMembership.role,
					},
				})
			} else {
				await tx.user.delete({ where: { id: userId } })
			}
		})
	}

	private async ensureUserMemberships(userId: string) {
		const user = await this.prisma.user.findUniqueOrThrow({ where: { id: userId } })
		await this.ensureDefaultMembership(user)
	}

	private async ensureDefaultMembership(user: User) {
		const existing = await this.prisma.organizationMember.findUnique({
			where: {
				userId_organizationId: {
					userId: user.id,
					organizationId: user.organizationId,
				},
			},
		})
		if (existing) return

		await this.prisma.organizationMember.create({
			data: {
				userId: user.id,
				organizationId: user.organizationId,
				role: user.role || 'admin',
			},
		})
	}

	private async requireOrganizationMember(userId: string, organizationId: string) {
		const membership = await this.prisma.organizationMember.findUnique({
			where: { userId_organizationId: { userId, organizationId } },
		})
		if (!membership) throw new ForbiddenException('You are not a member of this organization')
		return membership
	}

	private async requireOrganizationAdmin(userId: string, organizationId: string) {
		const membership = await this.requireOrganizationMember(userId, organizationId)
		if (membership.role !== 'admin') {
			throw new ForbiddenException('Only organization admins can perform this action')
		}
		return membership
	}

	private async createUniqueSlug(
		name: string,
		client: Pick<PrismaService, 'organization'> | Prisma.TransactionClient = this.prisma,
	) {
		const base = name
			.toLowerCase()
			.trim()
			.replace(/[^a-z0-9]+/g, '-')
			.replace(/^-+|-+$/g, '')
			.slice(0, 48)
		const safeBase = base || 'organization'
		let slug = safeBase
		let suffix = 1

		while (await client.organization.findUnique({ where: { slug } })) {
			slug = `${safeBase}-${suffix++}`
		}

		return slug
	}
}
