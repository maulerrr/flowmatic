import { Type } from 'class-transformer'
import { IsBoolean, IsIn, IsInt, IsObject, IsOptional, IsString, Max, MaxLength, Min, MinLength } from 'class-validator'
import { ExportAdapterType } from 'src/modules/export/types/export.types'

export class CreateExportTargetDto {
	@IsString()
	@MinLength(2)
	@MaxLength(120)
	name!: string

	@IsIn(['raw', 'cleaned', 'business'])
	stage!: 'raw' | 'cleaned' | 'business'

	@IsIn(Object.values(ExportAdapterType))
	adapterType!: ExportAdapterType

	@IsOptional()
	@IsObject()
	settings?: Record<string, unknown>

	@IsOptional()
	@IsBoolean()
	saveCredentials?: boolean

	@IsOptional()
	@IsBoolean()
	isContinuous?: boolean

	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(15)
	@Max(3600)
	cadenceSeconds?: number
}
