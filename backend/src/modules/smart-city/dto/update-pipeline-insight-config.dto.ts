import { IsIn, IsInt, IsOptional, Min } from 'class-validator'

export class UpdatePipelineInsightConfigDto {
	@IsOptional()
	@IsInt()
	@Min(0)
	@IsIn([0, 15, 30, 60, 360, 1440])
	intervalMinutes?: number

	@IsOptional()
	@IsIn(['quick', 'standard', 'deep'])
	depth?: 'quick' | 'standard' | 'deep'

	@IsOptional()
	@IsIn(['all', 'ops', 'quality', 'geo', 'exports'])
	focus?: 'all' | 'ops' | 'quality' | 'geo' | 'exports'
}
