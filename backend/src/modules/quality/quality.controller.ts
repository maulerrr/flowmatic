import { Controller, Post, Body, UseGuards } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { QualityService } from './quality.service'
import { AnalyzeQualityDto } from './dto/analyze-quality.dto'
import { AuthGuard } from '../auth/auth.guard'

@ApiTags('quality')
@Controller('quality')
@UseGuards(AuthGuard)
export class QualityController {
	constructor(private readonly qualityService: QualityService) {}

	@Post('analyze')
	analyzeQuality(@Body() body: AnalyzeQualityDto) {
		const report = this.qualityService.analyzeQuality(body.data, body.columns)
		return {
			success: true,
			message: 'Quality analysis completed',
			data: report,
		}
	}
}
