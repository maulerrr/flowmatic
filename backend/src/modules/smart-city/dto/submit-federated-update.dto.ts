import { IsInt, IsObject, IsOptional, IsString, Max, MaxLength, Min } from 'class-validator'
import { Type } from 'class-transformer'

export class SubmitFederatedUpdateDto {
	@IsOptional()
	@IsString()
	@MaxLength(512)
	checkpointUri?: string

	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	@Max(1000000000)
	sampleCount?: number

	@IsOptional()
	@IsObject()
	metrics?: Record<string, unknown>

	@IsOptional()
	@IsString()
	@MaxLength(500)
	notes?: string
}
