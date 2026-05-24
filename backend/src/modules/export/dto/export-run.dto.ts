import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger'
import { Type } from 'class-transformer'
import { IsBoolean, IsEnum, IsObject, IsOptional } from 'class-validator'
import { ExportAdapterType } from '../types/export.types'

export class ExportRunDto {
	@ApiProperty({ enum: ExportAdapterType })
	@IsEnum(ExportAdapterType)
	adapterType: ExportAdapterType

	@ApiProperty({ type: Object })
	@IsObject()
	settings: Record<string, unknown>

	@ApiPropertyOptional({ default: false })
	@IsOptional()
	@Type(() => Boolean)
	@IsBoolean()
	saveCredentials?: boolean = false
}
