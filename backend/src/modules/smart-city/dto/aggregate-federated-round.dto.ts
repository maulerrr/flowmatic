import { IsObject, IsOptional, IsString, MaxLength } from 'class-validator'

export class AggregateFederatedRoundDto {
	@IsOptional()
	@IsString()
	@MaxLength(255)
	globalModelVersion?: string

	@IsOptional()
	@IsString()
	@MaxLength(512)
	checkpointUri?: string

	@IsOptional()
	@IsObject()
	metrics?: Record<string, unknown>

	@IsOptional()
	@IsString()
	@MaxLength(500)
	summary?: string
}
