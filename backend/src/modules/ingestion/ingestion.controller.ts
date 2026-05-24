import { Controller, Post, UseGuards, BadRequestException, Req } from '@nestjs/common'
import { ApiTags, ApiConsumes, ApiBody } from '@nestjs/swagger'
import * as fs from 'fs/promises'
import * as path from 'path'
import { MultipartFile } from '@fastify/multipart'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { IngestionService } from './ingestion.service'
import { PipelineService } from '../pipeline/pipeline.service'
import { StorageService } from '../storage/storage.service'
import { AuthGuard } from '../auth/auth.guard'
import { PrismaService } from 'src/prisma/prisma.service'

@ApiTags('ingestion')
@Controller('ingestion')
@UseGuards(AuthGuard)
export class IngestionController {
	constructor(
		private readonly ingestionService: IngestionService,
		private readonly pipelineService: PipelineService,
		private readonly storageService: StorageService,
		private readonly prisma: PrismaService,
	) {}

	@Post('upload')
	@ApiConsumes('multipart/form-data')
	@ApiBody({
		schema: {
			type: 'object',
			properties: {
				file: {
					type: 'string',
					format: 'binary',
				},
			},
		},
	})
	async uploadFile(@Req() req: AuthenticatedRequest) {
		const file = await req.file()
		if (!file) {
			throw new BadRequestException('No file uploaded')
		}
		const upload = file as MultipartFile
		const fileBuffer = await upload.toBuffer()
		const originalName = upload.filename
		const mimetype = upload.mimetype

		const { organizationId, userId } = req.authContext!

		let tempFilePath: string | null = null

		try {
			// Create temporary file from buffer
			const tmpDir = path.join(process.cwd(), 'tmp')
			await fs.mkdir(tmpDir, { recursive: true })

			tempFilePath = path.join(tmpDir, `${Date.now()}_${originalName}`)
			await fs.writeFile(tempFilePath, fileBuffer)

			// Validate file by ingesting it
			const ingestedData = await this.ingestionService.ingestFromFile(tempFilePath)

			// Generate S3 key and upload to S3
			const s3Key = this.storageService.generateS3Key(organizationId, originalName, 'source')

			await this.storageService.uploadFileToS3({
				bucket: this.storageService.defaultBucket,
				key: s3Key,
				body: fileBuffer,
				contentType: mimetype,
				metadata: {
					organizationId,
					userId,
					originalName,
				},
			})

			// Create storage file record
			const storageFile = await this.prisma.storageFile.create({
				data: {
					organizationId,
					fileName: originalName,
					fileSize: fileBuffer.length,
					mimeType: mimetype,
					s3Key,
				},
			})

			// Create pipeline run (async processing)
			const { runId, jobId } = await this.pipelineService.createPipelineRun(
				organizationId,
				storageFile.id,
				originalName,
			)

			return {
				success: true,
				message: 'File uploaded and queued for processing',
				data: {
					runId,
					jobId,
					fileId: storageFile.id,
					fileName: originalName,
					fileSize: fileBuffer.length,
					preview: {
						columns: ingestedData.columns.slice(0, 10),
						rowCount: ingestedData.rowCount,
						sampleRows: ingestedData.data.slice(0, 3),
					},
				},
			}
		} finally {
			// Clean up temporary file
			if (tempFilePath) {
				try {
					await fs.unlink(tempFilePath)
				} catch {
					// Ignore cleanup errors
				}
			}
		}
	}
}
