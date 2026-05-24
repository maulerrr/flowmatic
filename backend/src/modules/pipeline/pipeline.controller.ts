import {
	Controller,
	Post,
	Get,
	Param,
	Delete,
	Query,
	UseGuards,
	Req,
	NotFoundException,
} from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { ParseCuidPipe } from 'src/common/pipes/parse-cuid.pipe'
import { PipelineService } from './pipeline.service'
import { AuthGuard } from '../auth/auth.guard'
import { ListRunsQueryDto } from './dto/list-runs-query.dto'
import { CleanupRunsQueryDto } from './dto/cleanup-runs-query.dto'

@ApiTags('pipelines')
@Controller('pipelines')
@UseGuards(AuthGuard)
export class PipelineController {
	constructor(private readonly pipelineService: PipelineService) {}

	@Get('analytics/summary')
	async getAnalyticsSummary(@Req() req: AuthenticatedRequest) {
		const stats = await this.pipelineService.getAnalyticsSummary(req.authContext!.organizationId)
		return {
			success: true,
			data: stats,
		}
	}

	@Get('analytics/charts')
	async getAnalyticsCharts(
		@Req() req: AuthenticatedRequest,
		@Query('period') period: string = '7d',
	) {
		const charts = await this.pipelineService.getAnalyticsCharts(
			req.authContext!.organizationId,
			period,
		)
		return {
			success: true,
			data: charts,
		}
	}

	@Get('runs')
	async listRuns(@Req() req: AuthenticatedRequest, @Query() query: ListRunsQueryDto) {
		const runs = await this.pipelineService.getPipelineRuns(
			req.authContext!.organizationId,
			query.limit,
			query.offset,
			query.status,
		)
		return {
			success: true,
			data: runs,
		}
	}

	@Delete('runs/:id')
	async deleteRun(@Param('id', ParseCuidPipe) id: string, @Req() req: AuthenticatedRequest) {
		const deleted = await this.pipelineService.deleteRun(id, req.authContext!.organizationId)
		if (!deleted) {
			throw new NotFoundException('Pipeline run not found')
		}

		return {
			success: true,
			message: 'Pipeline run deleted',
		}
	}

	@Get('runs/:id/preview')
	async getRunPreview(@Param('id', ParseCuidPipe) id: string, @Req() req: AuthenticatedRequest) {
		const preview = await this.pipelineService.getRunPreview(id, req.authContext!.organizationId)
		if (!preview) {
			throw new NotFoundException('Run or preview data not found')
		}

		return {
			success: true,
			data: preview,
		}
	}

	@Get('runs/:id')
	async getRun(@Param('id', ParseCuidPipe) id: string, @Req() req: AuthenticatedRequest) {
		const run = await this.pipelineService.getPipelineRun(id, req.authContext!.organizationId)
		if (!run) {
			throw new NotFoundException('Pipeline run not found')
		}

		return {
			success: true,
			data: run,
		}
	}

	@Post('cleanup')
	async cleanupOldRuns(@Req() req: AuthenticatedRequest, @Query() query: CleanupRunsQueryDto) {
		const deleted = await this.pipelineService.cleanupOldRuns(
			req.authContext!.organizationId,
			query.daysOld ?? 30,
			query.statuses ?? ['failed', 'completed'],
		)

		return {
			success: true,
			message: `Deleted ${deleted} old pipeline runs`,
			count: deleted,
		}
	}
}
