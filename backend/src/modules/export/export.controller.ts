import {
	Controller,
	Get,
	Post,
	Param,
	Body,
	UseGuards,
	Req,
	Query,
} from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { ExportService } from './export.service'
import { AuthGuard } from '../auth/auth.guard'
import { ParseCuidPipe } from 'src/common/pipes/parse-cuid.pipe'
import { PaginationParamsFilter } from 'src/common/utils/pagination.util'
import { ExportRunDto } from './dto/export-run.dto'
import { ValidateExportDto } from './dto/validate-export.dto'

@ApiTags('exports')
@Controller('exports')
@UseGuards(AuthGuard)
export class ExportController {
	constructor(private readonly exportService: ExportService) {}

	/**
	 * Get available export adapters and their configuration requirements
	 */
	@Get('adapters')
	getAvailableAdapters() {
		const adapters = this.exportService.getAvailableAdapters()
		return {
			success: true,
			data: adapters,
		}
	}

	/**
	 * Preview cleaned data from a pipeline run (shows sample rows)
	 * GET /api/v1/exports/runs/:runId/preview?page=1&pageSize=10
	 */
	@Get('runs/:runId/preview')
	async previewRunData(
		@Req() req: AuthenticatedRequest,
		@Param('runId', ParseCuidPipe) runId: string,
		@Query() query: PaginationParamsFilter,
	) {
		const result = await this.exportService.getPreviewData(
			runId,
			req.authContext!.organizationId,
			query,
		)

		return {
			success: true,
			data: result,
		}
	}

	/**
	 * Export pipeline run data to specified destination
	 * POST /api/v1/exports/runs/:runId/export
	 * Body: { adapterType: 'postgres' | 'mongodb' | 'huggingface' | 'csv' | 'json', settings: {...} }
	 */
	@Post('runs/:runId/export')
	async exportRunData(
		@Param('runId', ParseCuidPipe) runId: string,
		@Body() body: ExportRunDto,
		@Req() req: AuthenticatedRequest,
	) {
		const result = await this.exportService.exportPipelineRun(
			runId,
			req.authContext!.organizationId,
			body.adapterType,
			body.settings,
			body.saveCredentials,
		)

		return {
			success: true,
			data: result,
			message: result.message,
		}
	}

	/**
	 * Validate export configuration
	 * POST /api/v1/exports/validate
	 */
	@Post('validate')
	async validateExportConfig(@Body() body: ValidateExportDto) {
		const validation = await this.exportService.validateExportConfig({
			adapterType: body.adapterType,
			organizationId: '',
			pipelineRunId: '',
			fileName: '',
			settings: body.settings || {},
		})

		return {
			success: validation.valid,
			data: validation,
		}
	}

	/**
	 * Get export history for a pipeline run
	 */
	@Get('runs/:runId/history')
	async getExportHistory(
		@Param('runId', ParseCuidPipe) runId: string,
		@Req() req: AuthenticatedRequest,
	) {
		const history = await this.exportService.getExportHistory(
			runId,
			req.authContext!.organizationId,
		)

		return {
			success: true,
			data: history,
		}
	}
}
