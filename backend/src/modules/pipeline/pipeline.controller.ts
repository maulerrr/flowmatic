import { Controller, Post, Get, Param, Delete, Query, UseGuards, Req, BadRequestException, NotFoundException } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { Request } from 'express'
import { PipelineService } from './pipeline.service'
import { AuthGuard } from '../auth/auth.guard'
import { AuthContext } from '../auth/auth-context.service'

declare global {
	namespace Express {
		interface Request {
			authContext?: AuthContext
		}
	}
}

@ApiTags('pipelines')
@Controller('pipelines')
@UseGuards(AuthGuard)
export class PipelineController {
	constructor(private readonly pipelineService: PipelineService) {}

	@Get('analytics/summary')
	async getAnalyticsSummary(@Req() req: Request) {
		const stats = await this.pipelineService.getAnalyticsSummary(req.authContext!.organizationId)
		return {
			success: true,
			data: stats,
		}
	}

	@Get('analytics/charts')
	async getAnalyticsCharts(@Req() req: Request, @Query('period') period: string = '7d') {
		const charts = await this.pipelineService.getAnalyticsCharts(req.authContext!.organizationId, period)
		return {
			success: true,
			data: charts,
		}
	}

	@Get('runs')
	async listRuns(
		@Req() req: Request,
		@Query('limit') limit?: string,
		@Query('offset') offset?: string,
		@Query('status') status?: string,
	) {
		const limitNum = limit ? Math.min(parseInt(limit, 10), 100) : 50
		const offsetNum = offset ? parseInt(offset, 10) : 0

		const runs = await this.pipelineService.getPipelineRuns(
			req.authContext!.organizationId,
			limitNum,
			offsetNum,
			status,
		)
		return {
			success: true,
			data: runs,
		}
	}

	@Delete('runs/:id')
	async deleteRun(@Param('id') id: string, @Req() req: Request) {
		if (!id || id.length < 5) {
			throw new BadRequestException('Invalid run ID')
		}

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
	async getRunPreview(@Param('id') id: string, @Req() req: Request) {
		if (!id || id.length < 5) {
			throw new BadRequestException('Invalid run ID')
		}

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
	async getRun(@Param('id') id: string, @Req() req: Request) {
		if (!id || id.length < 5) {
			throw new BadRequestException('Invalid run ID')
		}

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
	async cleanupOldRuns(
		@Req() req: Request,
		@Query('daysOld') daysOld?: string,
		@Query('statuses') statuses?: string,
	) {
		const daysOldNum = daysOld ? parseInt(daysOld, 10) : 30
		const statusArray = statuses ? statuses.split(',') : ['failed', 'completed']

		if (daysOldNum < 1) {
			throw new BadRequestException('daysOld must be at least 1')
		}

		const deleted = await this.pipelineService.cleanupOldRuns(
			req.authContext!.organizationId,
			daysOldNum,
			statusArray,
		)

		return {
			success: true,
			message: `Deleted ${deleted} old pipeline runs`,
			count: deleted,
		}
	}
}
