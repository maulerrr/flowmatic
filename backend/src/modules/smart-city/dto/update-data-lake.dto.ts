import { PartialType } from '@nestjs/mapped-types'
import { CreateDataLakeDto } from './create-data-lake.dto'

export class UpdateDataLakeDto extends PartialType(CreateDataLakeDto) {}
