import { ApiProperty } from '@nestjs/swagger'
import { IsOptional, IsString, MinLength } from 'class-validator'

export class UpdateProfileDto {
	@ApiProperty({ required: false })
	@IsOptional()
	@IsString()
	@MinLength(1)
	displayName?: string
}
