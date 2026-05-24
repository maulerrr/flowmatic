import { ApiPropertyOptional } from '@nestjs/swagger'
import { Transform, Type } from 'class-transformer'
import { ArrayNotEmpty, IsArray, IsIn, IsInt, IsOptional, Min } from 'class-validator'

const CLEANUP_STATUSES = ['queued', 'processing', 'completed', 'failed'] as const

export class CleanupRunsQueryDto {
	@ApiPropertyOptional({ minimum: 1, default: 30 })
	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	daysOld?: number = 30

	@ApiPropertyOptional({ type: String, example: 'failed,completed' })
	@IsOptional()
	@Transform(({ value }) => {
		if (!value) return ['failed', 'completed']
		if (Array.isArray(value)) return value
		return String(value)
			.split(',')
			.map(item => item.trim())
			.filter(Boolean)
	})
	@IsArray()
	@ArrayNotEmpty()
	@IsIn(CLEANUP_STATUSES, { each: true })
	statuses?: string[] = ['failed', 'completed']
}
