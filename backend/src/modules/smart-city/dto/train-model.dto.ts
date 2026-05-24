import { IsIn, IsOptional, IsString, MaxLength } from 'class-validator'

export class TrainModelDto {
	@IsOptional()
	@IsString()
	@MaxLength(120)
	name?: string

	@IsOptional()
	@IsString()
	pipelineId?: string

	@IsOptional()
	@IsString()
	datasetPath?: string

	@IsOptional()
	@IsIn(['TRAFFIC_BASELINE', 'ANOMALY_BASELINE'])
	modelType?: string
}
