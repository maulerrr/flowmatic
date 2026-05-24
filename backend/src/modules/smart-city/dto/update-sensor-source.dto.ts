import { IsIn, IsInt, IsObject, IsOptional, IsString, Max, MaxLength, Min, MinLength } from 'class-validator'

export class UpdateSensorSourceDto {
	@IsOptional()
	@IsString()
	@MinLength(2)
	@MaxLength(120)
	name?: string

	@IsOptional()
	@IsIn(['WEBSOCKET', 'HTTP_POLLING'])
	type?: string

	@IsOptional()
	@IsString()
	@MaxLength(40)
	sensorKind?: string

	@IsOptional()
	@IsIn(['SIMULATED', 'EXTERNAL'])
	mode?: string

	@IsOptional()
	@IsString()
	@MaxLength(500)
	endpoint?: string

	@IsOptional()
	@IsInt()
	@Min(1000)
	@Max(600000)
	pollIntervalMs?: number

	@IsOptional()
	@IsObject()
	connectionConfig?: Record<string, unknown>
}
