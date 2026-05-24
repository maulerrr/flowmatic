import { IsBoolean, IsIn, IsObject, IsOptional, IsString, MaxLength, MinLength } from 'class-validator'

export class CreateDataLakeDto {
	@IsString()
	@MinLength(2)
	@MaxLength(120)
	name!: string

	@IsOptional()
	@IsIn(['AWS_S3', 'MINIO', 'R2', 'CUSTOM_S3'])
	provider?: string

	@IsString()
	@MinLength(2)
	@MaxLength(200)
	bucket!: string

	@IsOptional()
	@IsString()
	@MaxLength(120)
	region?: string

	@IsOptional()
	@IsString()
	@MaxLength(500)
	endpoint?: string

	@IsOptional()
	@IsString()
	@MaxLength(500)
	basePrefix?: string

	@IsOptional()
	@IsString()
	accessKey?: string

	@IsOptional()
	@IsString()
	secretKey?: string

	@IsOptional()
	@IsBoolean()
	isDefault?: boolean

	@IsOptional()
	@IsObject()
	pathRules?: Record<string, unknown>
}
