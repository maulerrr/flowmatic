import { ApiProperty } from '@nestjs/swagger'
import { IsEnum, IsObject } from 'class-validator'
import { ExportAdapterType } from '../types/export.types'

export class ValidateExportDto {
	@ApiProperty({ enum: ExportAdapterType })
	@IsEnum(ExportAdapterType)
	adapterType: ExportAdapterType

	@ApiProperty({ type: Object })
	@IsObject()
	settings: Record<string, unknown>
}
