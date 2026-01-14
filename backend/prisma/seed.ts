// prisma/seed.ts

import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
	console.log('🌱 Seeding database...')

	// Create default organization
	const org = await prisma.organization.upsert({
		where: { slug: 'flowmatic' },
		update: {},
		create: {
			name: 'Flowmatic',
			slug: 'flowmatic',
		},
	})
	console.log('✅ Created organization:', org.name)

	// Create default admin user
	const user = await prisma.user.upsert({
		where: { email: 'admin@flowmatic.local' },
		update: {},
		create: {
			email: 'admin@flowmatic.local',
			displayName: 'Admin User',
			organizationId: org.id,
			role: 'admin',
		},
	})
	console.log('✅ Created user:', user.email)

	console.log('🎉 Seeding complete!')
}

main()
	.catch((e) => {
		console.error(e)
		process.exit(1)
	})
	.finally(async () => {
		await prisma.$disconnect()
	})
