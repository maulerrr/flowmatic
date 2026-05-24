import { ApiProperty } from '@nestjs/swagger'
import { ArrayNotEmpty, IsArray, IsObject, IsString } from 'class-validator'
import { DataRow } from 'src/common/types/data.types'

export class AnalyzeQualityDto {
	@ApiProperty({ type: 'array', items: { type: 'object' } })
	@IsArray()
	@IsObject({ each: true })
	data: DataRow[]

	@ApiProperty({ type: [String] })
	@IsArray()
	@ArrayNotEmpty()
	@IsString({ each: true })
	columns: string[]
}
