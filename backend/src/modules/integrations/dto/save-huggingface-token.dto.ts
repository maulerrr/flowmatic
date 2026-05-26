import { ApiProperty } from '@nestjs/swagger'
import { IsString, MinLength } from 'class-validator'

export class SaveHuggingFaceTokenDto {
	@ApiProperty({ description: 'Hugging Face access token (hf_...)' })
	@IsString()
	@MinLength(8)
	token: string
}
