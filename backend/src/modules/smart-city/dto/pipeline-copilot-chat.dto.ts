import { IsOptional, IsString } from 'class-validator'

export class PipelineCopilotChatDto {
	@IsOptional()
	@IsString()
	chipId?: string

	@IsOptional()
	@IsString()
	message?: string
}
