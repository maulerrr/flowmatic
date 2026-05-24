import { ApiProperty } from '@nestjs/swagger'
import { IsEmail, IsIn, IsOptional } from 'class-validator'
import { UserRole } from '../auth-context.service'

export class InviteMemberDto {
	@ApiProperty()
	@IsEmail()
	email: string

	@ApiProperty({ required: false, enum: ['admin', 'member', 'viewer'] })
	@IsOptional()
	@IsIn(['admin', 'member', 'viewer'])
	role?: UserRole
}
