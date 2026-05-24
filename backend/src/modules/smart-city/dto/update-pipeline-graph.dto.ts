import { IsObject } from 'class-validator'

export class UpdatePipelineGraphDto {
	@IsObject()
	graph!: Record<string, unknown>
}
