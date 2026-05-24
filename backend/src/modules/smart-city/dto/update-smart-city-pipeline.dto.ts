import { IsIn, IsObject, IsOptional, IsString, MaxLength, MinLength } from 'class-validator'

export class UpdateSmartCityPipelineDto {
	@IsOptional()
	@IsString()
	@MinLength(2)
	@MaxLength(120)
	name?: string

	@IsOptional()
	@IsString()
	@MaxLength(1000)
	description?: string

	@IsOptional()
	@IsIn(['DRAFT', 'ACTIVE', 'PAUSED', 'ERROR', 'ARCHIVED'])
	status?: string

	@IsOptional()
	@IsObject()
	streamConfig?: Record<string, unknown>

	@IsOptional()
	@IsString()
	activeModelId?: string

	@IsOptional()
	@IsString()
	dataLakeConnectionId?: string
}
