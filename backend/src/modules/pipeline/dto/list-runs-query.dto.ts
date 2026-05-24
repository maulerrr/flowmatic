import { ApiPropertyOptional } from '@nestjs/swagger'
import { Type } from 'class-transformer'
import { IsIn, IsInt, IsOptional, Max, Min } from 'class-validator'

const PIPELINE_STATUSES = ['queued', 'processing', 'completed', 'failed'] as const

export class ListRunsQueryDto {
	@ApiPropertyOptional({ minimum: 1, maximum: 100, default: 50 })
	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	@Max(100)
	limit?: number = 50

	@ApiPropertyOptional({ minimum: 0, default: 0 })
	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(0)
	offset?: number = 0

	@ApiPropertyOptional({ enum: PIPELINE_STATUSES })
	@IsOptional()
	@IsIn(PIPELINE_STATUSES)
	status?: (typeof PIPELINE_STATUSES)[number]
}
