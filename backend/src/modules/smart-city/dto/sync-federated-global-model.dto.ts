import { IsBoolean, IsOptional } from 'class-validator'

export class SyncFederatedGlobalModelDto {
	@IsOptional()
	@IsBoolean()
	includeRounds?: boolean
}
