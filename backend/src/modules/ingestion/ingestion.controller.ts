import {
	Controller,
	Post,
	UploadedFile,
	UseInterceptors,
	UseGuards,
	BadRequestException,
	Req,
} from '@nestjs/common'
import { FileInterceptor } from '@nestjs/platform-express'
import { ApiTags, ApiConsumes, ApiBody } from '@nestjs/swagger'
import * as fs from 'fs/promises'
import * as path from 'path'
import { Request } from 'express'
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
	@UseInterceptors(FileInterceptor('file'))
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
	async uploadFile(@UploadedFile() file: Express.Multer.File, @Req() req: Request) {
		if (!file) {
			throw new BadRequestException('No file uploaded')
		}

		const { organizationId, userId } = req.authContext!

		let tempFilePath: string | null = null

		try {
			// Create temporary file from buffer
			const tmpDir = path.join(process.cwd(), 'tmp')
			await fs.mkdir(tmpDir, { recursive: true })

			tempFilePath = path.join(tmpDir, `${Date.now()}_${file.originalname}`)
			await fs.writeFile(tempFilePath, file.buffer)

			// Validate file by ingesting it
			const ingestedData = await this.ingestionService.ingestFromFile(tempFilePath)

			// Generate S3 key and upload to S3
			const s3Key = this.storageService.generateS3Key(organizationId, file.originalname, 'source')

			const s3Bucket = process.env.S3_BUCKET || 'flowmatic-uploads'
			await this.storageService.uploadFileToS3({
				bucket: s3Bucket,
				key: s3Key,
				body: file.buffer,
				contentType: file.mimetype,
				metadata: {
					organizationId,
					userId,
					originalName: file.originalname,
				},
			})

			// Create storage file record
			const storageFile = await this.prisma.storageFile.create({
				data: {
					organizationId,
					fileName: file.originalname,
					fileSize: file.size,
					mimeType: file.mimetype,
					s3Key,
				},
			})

			// Create pipeline run (async processing)
			const { runId, jobId } = await this.pipelineService.createPipelineRun(
				organizationId,
				storageFile.id,
				file.originalname,
			)

			return {
				success: true,
				message: 'File uploaded and queued for processing',
				data: {
					runId,
					jobId,
					fileId: storageFile.id,
					fileName: file.originalname,
					fileSize: file.size,
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
