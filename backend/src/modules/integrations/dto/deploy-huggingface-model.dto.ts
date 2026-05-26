import { ApiProperty } from '@nestjs/swagger'
import { IsString, Matches } from 'class-validator'

export class DeployHuggingFaceModelDto {
	@ApiProperty({ example: 'username/my-model' })
	@IsString()
	@Matches(/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/)
	modelId: string
}
