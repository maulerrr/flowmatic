import { Type } from 'class-transformer'
import { IsIn, IsInt, IsOptional, Max, Min } from 'class-validator'

export class BackfillPipelineDto {
	@IsOptional()
	@IsIn(['raw', 'cleaned', 'business', 'all'])
	stage?: 'raw' | 'cleaned' | 'business' | 'all'

	@IsOptional()
	@Type(() => Number)
	@IsInt()
	@Min(1)
	@Max(5000)
	limit?: number
}
