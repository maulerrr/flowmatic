export interface AuthTokens {
	accessToken: string
	refreshToken: string
}

export enum ADMIN_ROLE {
	ADMIN = 'ADMIN',
	TEACHER = 'TEACHER',
	HELPDESK = 'HELPDESK',
}

export interface AdminDto {
	id: number
	name: string
	phoneNumber: string
	role: ADMIN_ROLE
	createdAt: Date
	updatedAt: Date
}
