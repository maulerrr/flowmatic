import {
	Controller,
	Get,
	Post,
	Param,
	Body,
	UseGuards,
	Req,
	BadRequestException,
	Query,
} from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { Request } from 'express'
import { ExportService } from './export.service'
import { AuthGuard } from '../auth/auth.guard'
import { AuthContext } from '../auth/auth-context.service'
import { PrismaService } from 'src/prisma/prisma.service'
import { ExportAdapterType } from './types/export.types'

declare global {
	namespace Express {
		interface Request {
			authContext?: AuthContext
		}
	}
}

@ApiTags('exports')
@Controller('exports')
@UseGuards(AuthGuard)
export class ExportController {
	constructor(
		private readonly exportService: ExportService,
		private readonly prisma: PrismaService,
	) {}

	/**
	 * Get available export adapters and their configuration requirements
	 */
	@Get('adapters')
	async getAvailableAdapters(@Req() req: Request) {
		const adapters = this.exportService.getAvailableAdapters()
		return {
			success: true,
			data: adapters,
		}
	}

	/**
	 * Preview cleaned data from a pipeline run (shows sample rows)
	 * GET /api/v1/exports/runs/:runId/preview?limit=10
	 */
	@Get('runs/:runId/preview')
	async previewRunData(
		@Req() req: Request,
		@Param('runId') runId: string,
		@Query('limit') limit?: string,
	) {
		const limitNum = limit ? Math.min(parseInt(limit, 10), 100) : 10

		// Verify run exists and user has access
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: runId },
			include: { sourceFile: true, resultFile: true },
		})

		if (!run) {
			throw new BadRequestException('Pipeline run not found')
		}

		if (run.organizationId !== req.authContext!.organizationId) {
			throw new BadRequestException('Unauthorized to preview this run')
		}

		// Generate sample data (in production, load from result file)
		const sampleData = this.generateSampleData(
			run.rowsCleaned || run.rowsIngested || limitNum,
			limitNum,
		)

		return {
			success: true,
			data: {
				runId,
				fileName: run.sourceFileName,
				totalRows: run.rowsCleaned || run.rowsIngested,
				previewRows: limitNum,
				columns: sampleData.length > 0 ? Object.keys(sampleData[0]) : [],
				data: sampleData,
			},
		}
	}

	/**
	 * Export pipeline run data to specified destination
	 * POST /api/v1/exports/runs/:runId/export
	 * Body: { adapterType: 'postgres' | 'mongodb' | 'huggingface' | 'csv' | 'json', settings: {...} }
	 */
	@Post('runs/:runId/export')
	async exportRunData(
		@Param('runId') runId: string,
		@Body()
		body: {
			adapterType: ExportAdapterType
			settings: Record<string, any>
		},
		@Req() req: Request,
	) {
		if (!body.adapterType || !body.settings) {
			throw new BadRequestException('adapterType and settings are required')
		}

		// Verify run exists and user has access
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: runId },
		})

		if (!run) {
			throw new BadRequestException('Pipeline run not found')
		}

		if (run.organizationId !== req.authContext!.organizationId) {
			throw new BadRequestException('Unauthorized to export this run')
		}

		try {
			const result = await this.exportService.exportPipelineRun(
				runId,
				req.authContext!.organizationId,
				body.adapterType,
				body.settings,
			)

			return {
				success: true,
				data: result,
				message: result.message,
			}
		} catch (error) {
			throw new BadRequestException(error instanceof Error ? error.message : 'Export failed')
		}
	}

	/**
	 * Validate export configuration
	 * POST /api/v1/exports/validate
	 */
	@Post('validate')
	async validateExportConfig(
		@Body()
		body: {
			adapterType: ExportAdapterType
			settings: Record<string, any>
		},
	) {
		if (!body.adapterType) {
			throw new BadRequestException('adapterType is required')
		}

		const validation = await this.exportService.validateExportConfig({
			adapterType: body.adapterType,
			organizationId: '',
			pipelineRunId: '',
			fileName: '',
			settings: body.settings || {},
		})

		return {
			success: validation.valid,
			errors: validation.errors || [],
		}
	}

	/**
	 * Get export history for a pipeline run
	 */
	@Get('runs/:runId/history')
	async getExportHistory(@Param('runId') runId: string, @Req() req: Request) {
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: runId },
		})

		if (!run) {
			throw new BadRequestException('Pipeline run not found')
		}

		if (run.organizationId !== req.authContext!.organizationId) {
			throw new BadRequestException('Unauthorized')
		}

		const history = await this.exportService.getExportHistory(
			runId,
			req.authContext!.organizationId,
		)

		return {
			success: true,
			data: history,
		}
	}

	/**
	 * Generate sample data for preview
	 */
	private generateSampleData(totalRows: number, sampleSize: number): any[] {
		const data: any[] = []
		const step = Math.max(1, Math.floor(totalRows / sampleSize))

		for (let i = 0; i < sampleSize && i * step < totalRows; i++) {
			data.push({
				id: i * step + 1,
				value: (Math.random() * 100).toFixed(2),
				status: ['active', 'inactive', 'pending'][Math.floor(Math.random() * 3)],
				created_at: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
				quality_score: (Math.random() * 100).toFixed(2),
			})
		}

		return data
	}
}
