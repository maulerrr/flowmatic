import { Injectable } from '@nestjs/common'
import { PrismaService } from 'src/prisma/prisma.service'
import * as bcrypt from 'bcrypt'

export interface AuthContext {
	userId: string
	organizationId: string
	role: string
}

@Injectable()
export class AuthContextService {
	constructor(private readonly prisma: PrismaService) {}

	async validateSession(token: string): Promise<AuthContext | null> {
		const session = await this.prisma.session.findUnique({
			where: { token },
		})

		if (!session) {
			return null
		}

		// Get user separately to avoid Prisma type issues
		const user = await this.prisma.user.findUnique({
			where: { id: session.userId },
		})

		if (!user || session.expiresAt < new Date()) {
			return null
		}

		return {
			userId: user.id,
			organizationId: user.organizationId,
			role: user.role,
		}
	}

	async createSession(userId: string, expiryDays: number = 30): Promise<string> {
		const token = Buffer.from(`${userId}:${Date.now()}:${Math.random()}`).toString('base64')

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
		await this.prisma.session.delete({
			where: { token },
		}).catch(() => {
			// Session may already be deleted
		})
	}

	async getOrCreateUser(email: string): Promise<{ user: any; isNew: boolean }> {
		let user = await this.prisma.user.findUnique({
			where: { email },
		})

		if (user) {
			return { user, isNew: false }
		}

		// Create default org for new user
		const org = await this.prisma.organization.create({
			data: {
				name: `${email.split('@')[0]}'s Organization`,
				slug: `org-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
			},
		})

		user = await this.prisma.user.create({
			data: {
				email,
				displayName: email.split('@')[0],
				organizationId: org.id,
				role: 'admin',
			},
		})

		return { user, isNew: true }
	}

	async getUser(userId: string) {
		return this.prisma.user.findUnique({
			where: { id: userId },
			select: {
				id: true,
				email: true,
				displayName: true,
				organizationId: true,
				role: true,
				createdAt: true,
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

	async deleteAccount(userId: string, organizationId: string): Promise<void> {
		// Used to cascade delete everything by deleting the organization
		// Ensure the user belongs to this organization (sanity check)
		const user = await this.prisma.user.findFirst({
			where: { id: userId, organizationId },
		})

		if (user) {
			await this.prisma.organization.delete({
				where: { id: organizationId },
			})
		}
	}
}
