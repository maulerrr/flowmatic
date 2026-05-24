import { Controller, Post, Body, UseGuards } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { CleaningService } from './cleaning.service'
import { CleanDataDto } from './dto/clean-data.dto'
import { AuthGuard } from '../auth/auth.guard'

@ApiTags('cleaning')
@Controller('cleaning')
@UseGuards(AuthGuard)
export class CleaningController {
	constructor(private readonly cleaningService: CleaningService) {}

	@Post('clean')
	cleanData(@Body() body: CleanDataDto) {
		const result = this.cleaningService.clean(
			body.data,
			body.numericColumns,
			body.categoricalColumns,
		)
		return {
			success: true,
			message: 'Data cleaned successfully',
			data: result,
		}
	}
}
