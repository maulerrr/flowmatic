import { Type } from 'class-transformer'
import { IsBoolean, IsIn, IsInt, IsObject, IsOptional, Max, Min } from 'class-validator'
import { ExportAdapterType } from 'src/modules/export/types/export.types'

export class ExportPipelineStageDto {
	@IsIn(Object.values(ExportAdapterType))
	adapterType!: ExportAdapterType

	@IsIn(['raw', 'cleaned', 'business'])
	stage!: 'raw' | 'cleaned' | 'business'

	@IsOptional()
	@IsObject()
	settings?: Record<string, unknown>

	@IsOptional()
	@IsBoolean()
	saveCredentials?: boolean

	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	@Max(500)
	limit?: number
}
