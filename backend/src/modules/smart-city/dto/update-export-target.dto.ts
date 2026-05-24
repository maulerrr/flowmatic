import { PartialType } from '@nestjs/mapped-types'
import { CreateExportTargetDto } from './create-export-target.dto'

export class UpdateExportTargetDto extends PartialType(CreateExportTargetDto) {}
