import { IsIn, IsInt, IsOptional, Max, Min } from 'class-validator'

export class UpdatePipelineRuntimeDto {
	@IsOptional()
	@IsInt()
	@Min(1000)
	@Max(3600000)
	sourcePollIntervalMs?: number

	@IsOptional()
	@IsIn(['append', 'object'])
	lakeWriteMode?: 'append' | 'object'

	@IsOptional()
	@IsInt()
	@Min(15)
	@Max(3600)
	exportCadenceSeconds?: number
}

export class StartPipelineDto {
	@IsOptional()
	@IsInt()
	@Min(1000)
	@Max(3600000)
	sourcePollIntervalMs?: number
}
