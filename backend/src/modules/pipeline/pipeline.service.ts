import { Inject, Injectable, Logger, OnModuleInit } from '@nestjs/common'
import { PipelineRun, StorageFile } from '@prisma/client'
import { ChatOpenAI } from '@langchain/openai'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { PrismaService } from 'src/prisma/prisma.service'
import { QualityReport, QualityService } from '../quality/quality.service'
import { CleaningService } from '../cleaning/cleaning.service'
import { StorageService } from '../storage/storage.service'
import { QUEUE_CLIENT, QueueClient, QueueMessage } from 'src/common/queue/queue.tokens'
import { PipelineJobData, PipelineJobResult, PipelineSummary } from './pipeline.types'
import { v4 as uuid } from 'uuid'
import { DataRow } from '../ingestion/ingestion.service'

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
				const errorMessage = error instanceof Error ? error.message : String(error)
				this.logger.error(`Failed to process pipeline job: ${errorMessage}`, error)
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
	): Promise<unknown[]> {
		const where: Record<string, unknown> = { organizationId }
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

	async getPipelineRun(
		runId: string,
		organizationId: string,
	): Promise<
		(PipelineRun & { sourceFile: StorageFile | null; resultFile: StorageFile | null }) | null
	> {
		return this.prisma.pipelineRun.findFirst({
			where: { id: runId, organizationId },
			include: {
				sourceFile: true,
				resultFile: true,
			},
		})
	}

	async getRunPreview(runId: string, organizationId: string): Promise<unknown> {
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
	private parseCsvBuffer(buffer: Buffer): { rows: DataRow[]; columns: string[] } {
		const text = buffer.toString('utf-8').trim()
		if (!text) return { rows: [], columns: [] }

		const lines = text.split(/\r?\n/).filter(l => l.length > 0)
		if (lines.length === 0) return { rows: [], columns: [] }

		const columns = this.parseCsvLine(lines[0])
		const rows: DataRow[] = []

		for (let i = 1; i < lines.length; i++) {
			const values = this.parseCsvLine(lines[i])
			if (values.length === 0) continue
			const row: DataRow = {}
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

	private async generateLlmSummary(
		fileName: string,
		ingested: number,
		cleaned: number,
		errors: number,
		quality: QualityReport,
	): Promise<string> {
		this.logger.log(`Generating summary for ${fileName} (ingested: ${ingested}, errors: ${errors})`)

		const numericCols = quality.numericColumns?.length || 0
		const categoricalCols = quality.categoricalColumns?.length || 0
		const errorRate = ingested > 0 ? (errors / ingested) * 100 : 0
		const cleanRate = ingested > 0 ? (cleaned / ingested) * 100 : 0

		// Fallback logic for when LLM is unavailable or fails
		const fallbackLogic = () => {
			this.logger.log('Executing fallback summary logic')
			// Calculate scores
			const initialScore = Math.round(Math.max(0, 100 - errorRate * 3))
			const finalScore = Math.min(100, Math.round(initialScore + (100 - initialScore) * 0.8)) // Simulate improvement

			let overview = `Analysis of the ${ingested.toLocaleString()} records from ${fileName}. `

			if (cleanRate > 98) {
				overview += `The dataset is highly clean, with ${cleaned.toLocaleString()} records successfully processed. `
			} else if (cleanRate > 80) {
				overview += `The dataset is generally composed of valid data, though ${errors.toLocaleString()} records required attention. `
			} else {
				overview += `Significant data quality issues were found, with a high rejection rate of ${errorRate.toFixed(1)}%. `
			}

			const insights = [
				`Identified ${numericCols} numeric fields and ${categoricalCols} categorical fields.`,
			]

			if (errors > 0) {
				insights.push(
					`${errors} records were flagged as potentially anomalous or containing schema violations.`,
				)
			} else {
				insights.push(`No significant schema violations or anomalies were detected.`)
			}

			const recommendation =
				errorRate > 10
					? 'Review source generation process for schema compliance.'
					: 'Proceed with downstream analytics.'

			return JSON.stringify({
				overview,
				scores: { initial: initialScore, final: finalScore },
				insights,
				recommendation,
			})
		}

		if (!process.env.OPENAI_API_KEY) {
			this.logger.warn('OPENAI_API_KEY not found, using fallback summary generation')
			return fallbackLogic()
		}

		try {
			this.logger.log('Attempting LLM summary generation with OpenAI')
			const chat = new ChatOpenAI({
				modelName: 'gpt-4o',
				temperature: 0.2, // Low temperature for consistent, structured output
				openAIApiKey: process.env.OPENAI_API_KEY,
			})

			const systemPrompt = `
You are an expert Data Quality Engineer. Your task is to analyze pipeline execution statistics and generate a JSON summary of data quality.

Output Format:
The output must be a valid JSON object with the following structure:
{
  "overview": "A concise narrative string summarizing the data quality and processing results.",
  "scores": { 
      "initial": number (0-100, estimate based on error rate), 
      "final": number (0-100, estimate after cleaning) 
  },
  "insights": ["string array of 2-3 key technical findings"],
  "recommendation": "A single string action item."
}

Scoring Rules:
- Initial Score: Roughly (100 - (error_rate * 3)). Penalize heavy errors.
- Final Score: Should be higher than initial, assuming cleaning fixed issues.
- Be professional but direct.

Few-Shot Example:
Input: "File: customer_data.csv. Ingested: 1000. Cleaned: 990. Errors: 10. Numeric Cols: 2. Categorical Cols: 3."
Output: {
  "overview": "Analysis of the 1,000 records from customer_data.csv. The dataset is highly clean, with 990 records successfully processed.",
  "scores": { "initial": 97, "final": 99 },
  "insights": [
    "Identified 2 numeric fields and 3 categorical fields.",
    "10 records were flagged as potentially anomalous or containing schema violations."
  ],
  "recommendation": "Proceed with downstream analytics."
}
`

			const userPrompt = `
Analyze this pipeline run:
File: ${fileName}
Ingested: ${ingested}
Cleaned: ${cleaned}
Errors: ${errors}
Numeric Columns: ${numericCols}
Categorical Columns: ${categoricalCols}
`

			const response = await chat.invoke([
				new SystemMessage(systemPrompt),
				new HumanMessage(userPrompt),
			])

			this.logger.log('LLM response received')
			let jsonStr = response.content as string
			// Clean markdown code blocks if present
			jsonStr = jsonStr.replace(/```json\n?|```/g, '').trim()

			// Validate JSON
			const summary = JSON.parse(jsonStr) as PipelineSummary

			// Basic validation of fields
			if (!summary.overview || !summary.scores || !summary.insights) {
				throw new Error('Invalid LLM summary format')
			}

			return jsonStr
		} catch (error) {
			this.logger.error('LLM Summary generation failed, falling back to mock:', error)
			return fallbackLogic()
		}
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
						const val: unknown = row[col]
						if (val === null || val === undefined) return ''
						const str = String(val as string | number | boolean | bigint | symbol)
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

			// Generate LLM Summary
			const summary = await this.generateLlmSummary(
				run.sourceFileName,
				rowsIngested,
				rowsCleaned,
				rowsErrors,
				quality,
			)

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
					summary,
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
