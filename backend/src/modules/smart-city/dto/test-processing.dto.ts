import { IsObject, IsOptional } from 'class-validator'

export class TestProcessingDto {
	@IsOptional()
	@IsObject()
	payload?: Record<string, unknown>
}
