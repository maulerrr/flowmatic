import { Controller, Post, Body } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { QualityService } from './quality.service'
import { DataRow } from '../ingestion/ingestion.service'

@ApiTags('quality')
@Controller('quality')
export class QualityController {
	constructor(private readonly qualityService: QualityService) {}

	@Post('analyze')
	analyzeQuality(@Body('data') data: DataRow[], @Body('columns') columns: string[]) {
		const report = this.qualityService.analyzeQuality(data, columns)
		return {
			success: true,
			message: 'Quality analysis completed',
			report,
		}
	}
}
