import { Inject, Injectable, Logger, OnModuleInit } from '@nestjs/common'
import { PrismaService } from 'src/prisma/prisma.service'
import { QualityService } from '../quality/quality.service'
import { CleaningService } from '../cleaning/cleaning.service'
import { StorageService } from '../storage/storage.service'
import { QUEUE_CLIENT, QueueClient, QueueMessage } from 'src/common/queue/queue.tokens'
import { PipelineJobData, PipelineJobResult } from './pipeline.types'
import { v4 as uuid } from 'uuid'

@Injectable()
export class PipelineService implements OnModuleInit {
	private readonly logger = new Logger(PipelineService.name)

	constructor(
		private prisma: PrismaService,
		private qualityService: QualityService,
		private cleaningService: CleaningService,
		private storageService: StorageService,
		@Inject(QUEUE_CLIENT) private queueClient: QueueClient,
	) {}

	async onModuleInit() {
		// Subscribe to pipeline queue messages
		await this.queueClient.subscribe<PipelineJobData>('pipeline', async msg => {
			try {
				await this.processPipelineJob(msg.payload)
			} catch (error) {
				this.logger.error(`Failed to process pipeline job: ${error.message}`, error)
			}
		})
		this.logger.log('Pipeline queue subscriber initialized')
	}

	async createPipelineRun(
		organizationId: string,
		sourceFileId: string,
		sourceFileName: string,
	): Promise<{ runId: string; jobId: string }> {
		// Create pipeline run record
		const run = await this.prisma.pipelineRun.create({
			data: {
				organizationId,
				sourceFileId,
				sourceFileName,
				status: 'queued',
			},
		})

		// Queue the job for async processing
		const jobId = uuid()
		const message: QueueMessage<PipelineJobData> = {
			id: jobId,
			type: 'pipeline.process',
			payload: {
				runId: run.id,
				organizationId,
				sourceFileId,
				sourceFileName,
			},
			occurredAt: new Date().toISOString(),
			version: 1,
		}

		await this.queueClient.publish('pipeline', message)

		// Store job ID for tracking
		await this.prisma.pipelineRun.update({
			where: { id: run.id },
			data: { jobId },
		})

		this.logger.log(`Pipeline run queued: ${run.id} (Job: ${jobId})`)

		return {
			runId: run.id,
			jobId,
		}
	}

	async getPipelineRuns(
		organizationId: string,
		limit: number = 50,
		offset: number = 0,
		status?: string,
	): Promise<any[]> {
		const where: any = { organizationId }
		if (status && status.length > 0) {
			where.status = status
		}

		return this.prisma.pipelineRun.findMany({
			where,
			include: {
				sourceFile: true,
				resultFile: true,
			},
			orderBy: { createdAt: 'desc' },
			take: limit,
			skip: offset,
		})
	}

	async getPipelineRun(runId: string, organizationId: string): Promise<any> {
		return this.prisma.pipelineRun.findFirst({
			where: { id: runId, organizationId },
			include: {
				sourceFile: true,
				resultFile: true,
			},
		})
	}

	async getRunPreview(runId: string, organizationId: string): Promise<any> {
		const run = await this.prisma.pipelineRun.findFirst({
			where: { id: runId, organizationId },
			include: {
				sourceFile: true,
			},
		})

		if (!run || !run.sourceFile) {
			return null
		}

		try {
			// Download source file from S3 and get preview
			const fileBuffer = await this.storageService.downloadFileFromS3(run.sourceFile.s3Key)
			const { rows, columns } = this.parseCsvBuffer(fileBuffer)
			const previewRows = rows.slice(0, 20)

			return {
				runId: run.id,
				fileName: run.sourceFileName,
				fileSize: run.sourceFile.fileSize,
				status: run.status,
				columns,
				preview: previewRows,
				stats: {
					rowsIngested: rows.length,
					rowsCleaned: run.rowsCleaned,
					rowsErrors: run.rowsErrors,
					processingTimeMs: run.processingTimeMs,
				},
				createdAt: run.createdAt,
				updatedAt: run.updatedAt,
			}
		} catch (error) {
			this.logger.error(`Failed to generate preview for run ${runId}:`, error)
			return null
		}
	}

	async deleteRun(runId: string, organizationId: string): Promise<boolean> {
		const run = await this.prisma.pipelineRun.findFirst({
			where: { id: runId, organizationId },
			include: { sourceFile: true, resultFile: true },
		})

		if (!run) {
			return false
		}

		try {
			const s3Bucket = process.env.S3_BUCKET || 'flowmatic-uploads'

			// Delete S3 files if they exist
			if (run.sourceFile?.s3Key) {
				await this.storageService.deleteFileFromS3(s3Bucket, run.sourceFile.s3Key).catch(() => {})
			}
			if (run.resultFile?.s3Key) {
				await this.storageService.deleteFileFromS3(s3Bucket, run.resultFile.s3Key).catch(() => {})
			}

			// Delete database records
			if (run.sourceFileId) {
				await this.prisma.storageFile.delete({ where: { id: run.sourceFileId } }).catch(() => {})
			}
			if (run.resultFileId) {
				await this.prisma.storageFile.delete({ where: { id: run.resultFileId } }).catch(() => {})
			}

			// Delete pipeline run
			await this.prisma.pipelineRun.delete({ where: { id: runId } })

			this.logger.log(`Pipeline run deleted: ${runId}`)
			return true
		} catch (error) {
			this.logger.error(`Failed to delete run ${runId}:`, error)
			throw error
		}
	}

	async cleanupOldRuns(
		organizationId: string,
		daysOld: number,
		statuses: string[],
	): Promise<number> {
		const cutoffDate = new Date(Date.now() - daysOld * 24 * 60 * 60 * 1000)

		const oldRuns = await this.prisma.pipelineRun.findMany({
			where: {
				organizationId,
				createdAt: { lt: cutoffDate },
				status: { in: statuses },
			},
			include: { sourceFile: true, resultFile: true },
		})

		if (oldRuns.length === 0) {
			return 0
		}

		// Delete in batches to avoid overwhelming the system
		let deleted = 0
		for (const run of oldRuns) {
			try {
				await this.deleteRun(run.id, organizationId)
				deleted++
			} catch (error) {
				this.logger.warn(`Failed to delete old run ${run.id}:`, error)
			}
		}

		this.logger.log(`Cleaned up ${deleted} old pipeline runs (${daysOld} days)`)
		return deleted
	}

	// Minimal CSV parser for pipeline processing and preview
	private parseCsvBuffer(buffer: Buffer): { rows: any[]; columns: string[] } {
		const text = buffer.toString('utf-8').trim()
		if (!text) return { rows: [], columns: [] }

		const lines = text.split(/\r?\n/).filter(l => l.length > 0)
		if (lines.length === 0) return { rows: [], columns: [] }

		const columns = this.parseCsvLine(lines[0])
		const rows: any[] = []

		for (let i = 1; i < lines.length; i++) {
			const values = this.parseCsvLine(lines[i])
			if (values.length === 0) continue
			const row: any = {}
			columns.forEach((col, idx) => {
				row[col] = values[idx] ?? ''
			})
			rows.push(row)
		}

		return { rows, columns }
	}

	private parseCsvLine(line: string): string[] {
		const result: string[] = []
		let current = ''
		let inQuotes = false

		for (let i = 0; i < line.length; i++) {
			const char = line[i]
			if (char === '"') {
				if (inQuotes && line[i + 1] === '"') {
					current += '"'
					i++
				} else {
					inQuotes = !inQuotes
				}
			} else if (char === ',' && !inQuotes) {
				result.push(current)
				current = ''
			} else {
				current += char
			}
		}
		result.push(current)
		return result
	}

	async processPipelineJob(data: PipelineJobData): Promise<PipelineJobResult> {
		const startTime = Date.now()
		const run = await this.prisma.pipelineRun.findUniqueOrThrow({
			where: { id: data.runId },
			include: { sourceFile: true },
		})

		if (!run.sourceFile) {
			throw new Error('Source file not found for pipeline run')
		}

		try {
			// Update status to processing
			await this.prisma.pipelineRun.update({
				where: { id: data.runId },
				data: { status: 'processing' },
			})

			this.logger.log(`Processing pipeline job: ${data.runId}`)

			// Download source file from S3
			const s3Bucket = process.env.S3_BUCKET || 'flowmatic-uploads'
			const fileBuffer = await this.storageService.downloadFileFromS3(
				run.sourceFile.s3Key,
				s3Bucket,
			)

			// Parse CSV data
			const { rows: rawRows, columns } = this.parseCsvBuffer(fileBuffer)
			const rowsIngested = rawRows.length

			// Run quality checks
			const quality = this.qualityService.analyzeQuality(rawRows, columns)

			// Run cleaning operations
			const cleaned = this.cleaningService.clean(
				rawRows,
				quality.numericColumns,
				quality.categoricalColumns,
			)
			const rowsCleaned = cleaned.data.length
			const rowsErrors = Math.max(0, rowsIngested - rowsCleaned)

			// Convert cleaned data back to CSV
			const headerLine = columns.join(',')
			const dataLines = cleaned.data.map(row =>
				columns
					.map(col => {
						const val = row[col]
						if (val === null || val === undefined) return ''
						const str = String(val)
						return str.includes(',') || str.includes('"') || str.includes('\n')
							? `"${str.replace(/"/g, '""')}"`
							: str
					})
					.join(','),
			)
			const resultCsv = [headerLine, ...dataLines].join('\n')
			const resultBuffer = Buffer.from(resultCsv, 'utf-8')

			// Upload result if cleaning succeeded
			let resultFileId: string | null = null
			if (resultBuffer && rowsCleaned > 0) {
				try {
					const resultS3Key = this.storageService.generateS3Key(
						data.organizationId,
						`${run.sourceFileName}_cleaned`,
						'results',
					)

					await this.storageService.uploadFileToS3({
						bucket: s3Bucket,
						key: resultS3Key,
						body: resultBuffer,
						contentType: 'text/csv',
						metadata: {
							organizationId: data.organizationId,
							pipelineRunId: data.runId,
						},
					})

					// Create result file record
					const resultFile = await this.prisma.storageFile.create({
						data: {
							organizationId: data.organizationId,
							fileName: `${run.sourceFileName}_cleaned`,
							fileSize: resultBuffer.length,
							mimeType: 'text/csv',
							s3Key: resultS3Key,
						},
					})

					resultFileId = resultFile.id
				} catch (error) {
					this.logger.error(`Failed to upload result file: ${error}`)
				}
			}

			const processingTimeMs = Date.now() - startTime

			// Update with completion status
			await this.prisma.pipelineRun.update({
				where: { id: data.runId },
				data: {
					status: 'completed',
					rowsIngested,
					rowsCleaned,
					rowsErrors,
					processingTimeMs,
					resultFileId,
				},
			})

			this.logger.log(
				`Pipeline job completed: ${data.runId} (ingested: ${rowsIngested}, cleaned: ${rowsCleaned}, errors: ${rowsErrors})`,
			)

			return {
				runId: data.runId,
				rowsIngested,
				rowsCleaned,
				rowsErrors,
				processingTimeMs,
			}
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Unknown error'

			await this.prisma.pipelineRun.update({
				where: { id: data.runId },
				data: {
					status: 'failed',
					errorMessage,
				},
			})

			this.logger.error(`Pipeline job failed: ${data.runId} - ${errorMessage}`, error)

			return {
				runId: data.runId,
				rowsIngested: 0,
				rowsCleaned: 0,
				rowsErrors: 0,
				processingTimeMs: Date.now() - startTime,
				errorMessage,
			}
		}
	}

	async getAnalyticsSummary(organizationId: string) {
		const runs = await this.prisma.pipelineRun.findMany({
			where: { organizationId },
			select: {
				status: true,
				processingTimeMs: true,
				sourceFile: {
					select: {
						fileSize: true,
					},
				},
			},
		})

		const total = runs.length
		const completed = runs.filter(r => r.status === 'completed').length
		const failed = runs.filter(r => r.status === 'failed').length
		const inProgress = runs.filter(r => r.status === 'processing' || r.status === 'queued').length

		const successRate = total > 0 ? (completed / total) * 100 : 0
		const totalDataProcessed = runs.reduce((sum, r) => sum + (r.sourceFile?.fileSize || 0), 0)

		const completedRuns = runs.filter(r => r.status === 'completed' && r.processingTimeMs > 0)
		const avgProcessingTime =
			completedRuns.length > 0
				? completedRuns.reduce((sum, r) => sum + r.processingTimeMs, 0) / completedRuns.length
				: 0

		return {
			total,
			completed,
			failed,
			inProgress,
			successRate,
			totalDataProcessed,
			avgProcessingTime,
		}
	}

	async getAnalyticsCharts(organizationId: string, period: string) {
		const now = new Date()
		let days = 7
		if (period === '30d') days = 30
		if (period === '90d') days = 90
		if (period === 'all') days = 365

		const startDate = new Date()
		startDate.setDate(now.getDate() - days)

		const runs = await this.prisma.pipelineRun.findMany({
			where: {
				organizationId,
				createdAt: {
					gte: startDate,
				},
			},
			orderBy: { createdAt: 'asc' },
			select: {
				createdAt: true,
				status: true,
			},
		})

		// Group by day
		const dailyCounts: Record<string, number> = {}
		runs.forEach(run => {
			const day = run.createdAt.toISOString().split('T')[0]
			dailyCounts[day] = (dailyCounts[day] || 0) + 1
		})

		const labels = Object.keys(dailyCounts)
		const values = Object.values(dailyCounts)

		return {
			uploadTrends: {
				labels,
				values,
			},
			statusDistribution: {
				labels: ['Completed', 'Failed', 'Processing'],
				values: [
					runs.filter(r => r.status === 'completed').length,
					runs.filter(r => r.status === 'failed').length,
					runs.filter(r => ['processing', 'queued'].includes(r.status)).length,
				],
			},
		}
	}
}
