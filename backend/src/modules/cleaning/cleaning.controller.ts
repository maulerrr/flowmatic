import { Controller, Post, Body } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { CleaningService } from './cleaning.service'
import { DataRow } from '../ingestion/ingestion.service'

@ApiTags('cleaning')
@Controller('cleaning')
export class CleaningController {
	constructor(private readonly cleaningService: CleaningService) {}

	@Post('clean')
	cleanData(
		@Body('data') data: DataRow[],
		@Body('numericColumns') numericColumns: string[],
		@Body('categoricalColumns') categoricalColumns: string[],
	) {
		const result = this.cleaningService.clean(data, numericColumns, categoricalColumns)
		return {
			success: true,
			message: 'Data cleaned successfully',
			result,
		}
	}
}
