import { IsIn, IsObject, IsOptional, IsString, MaxLength } from 'class-validator'

export class ConnectFederatedDto {
	@IsString()
	@MaxLength(500)
	endpoint!: string

	@IsIn(['HTTP', 'WEBSOCKET'])
	protocol!: 'HTTP' | 'WEBSOCKET'

	@IsOptional()
	@IsString()
	@MaxLength(255)
	projectId?: string

	@IsOptional()
	@IsString()
	@MaxLength(255)
	nodeId?: string

	@IsOptional()
	@IsString()
	@MaxLength(255)
	topic?: string

	@IsOptional()
	@IsString()
	@MaxLength(2048)
	apiKey?: string

	@IsOptional()
	@IsObject()
	headers?: Record<string, string>

	@IsOptional()
	@IsObject()
	registerPayload?: Record<string, unknown>
}
