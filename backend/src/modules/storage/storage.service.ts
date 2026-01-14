import { Injectable, Logger } from '@nestjs/common'
import { PrismaService } from 'src/prisma/prisma.service'
import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'

export interface S3UploadOptions {
	bucket: string
	key: string
	body: Buffer | string
	contentType?: string
	metadata?: Record<string, string>
}

@Injectable()
export class StorageService {
	private readonly logger = new Logger(StorageService.name)
	private s3Client: S3Client

	constructor(private prisma: PrismaService) {
		// Initialize S3 client only if credentials are provided
		const accessKey = process.env.S3_ACCESS_KEY || process.env.AWS_ACCESS_KEY_ID
		const secretKey = process.env.S3_SECRET_KEY || process.env.AWS_SECRET_ACCESS_KEY
		const endpoint = process.env.S3_ACCESS_ENDPOINT

		if (accessKey && secretKey) {
			const clientConfig: any = {
				region: process.env.S3_REGION || 'us-east-1',
				credentials: {
					accessKeyId: accessKey,
					secretAccessKey: secretKey,
				},
			}

			// Add endpoint if using MinIO or S3-compatible service
			if (endpoint) {
				clientConfig.endpoint = endpoint
				clientConfig.forcePathStyle = process.env.S3_PATH_STYLE === 'true'
			}

			this.s3Client = new S3Client(clientConfig)
		}
	}

	async uploadFileToS3(options: S3UploadOptions): Promise<string> {
		if (!this.s3Client) {
			throw new Error('S3 credentials not configured')
		}

		try {
			const command = new PutObjectCommand({
				Bucket: options.bucket,
				Key: options.key,
				Body: options.body,
				ContentType: options.contentType || 'application/octet-stream',
				Metadata: options.metadata,
			})

			await this.s3Client.send(command)
			this.logger.debug(`File uploaded to S3: s3://${options.bucket}/${options.key}`)

			return options.key
		} catch (error) {
			this.logger.error(`Failed to upload file to S3: ${error}`)
			throw error
		}
	}   

	async getSignedDownloadUrl(bucket: string, key: string, expiresIn: number = 3600): Promise<string> {
		if (!this.s3Client) {
			throw new Error('S3 credentials not configured')
		}

		try {
			const command = new GetObjectCommand({
				Bucket: bucket,
				Key: key,
			})

			const signedUrl = await getSignedUrl(this.s3Client, command, { expiresIn })
			return signedUrl
		} catch (error) {
			this.logger.error(`Failed to generate signed URL: ${error}`)
			throw error
		}
	}

	async deleteFileFromS3(bucket: string, key: string): Promise<void> {
		if (!this.s3Client) {
			throw new Error('S3 credentials not configured')
		}

		try {
			const command = new DeleteObjectCommand({
				Bucket: bucket,
				Key: key,
			})

			await this.s3Client.send(command)
			this.logger.debug(`File deleted from S3: s3://${bucket}/${key}`)
		} catch (error) {
			this.logger.error(`Failed to delete file from S3: ${error}`)
			throw error
		}
	}

	async downloadFileFromS3(key: string, bucket?: string): Promise<Buffer> {
		if (!this.s3Client) {
			throw new Error('S3 credentials not configured')
		}

		const s3Bucket = bucket || process.env.S3_BUCKET || 'flowmatic-uploads'

		try {
			const command = new GetObjectCommand({
				Bucket: s3Bucket,
				Key: key,
			})

			const response = await this.s3Client.send(command)
			const chunks: Buffer[] = []

			// Convert ReadableStream to Buffer
			if (response.Body instanceof Buffer) {
				return response.Body
			}

			// Handle stream response
			if (response.Body && 'on' in response.Body) {
				const stream = response.Body as any
				return new Promise((resolve, reject) => {
					stream.on('data', (chunk: Buffer) => chunks.push(chunk))
					stream.on('end', () => resolve(Buffer.concat(chunks)))
					stream.on('error', reject)
				})
			}

			throw new Error('Unexpected response body type from S3')
		} catch (error) {
			this.logger.error(`Failed to download file from S3 (key: ${key}):`, error)
			throw error
		}
	}

	generateS3Key(organizationId: string, fileName: string, prefix: string = 'uploads'): string {
		const timestamp = Date.now()
		const random = Math.random().toString(36).slice(2, 9)
		const safeFileName = fileName.replace(/[^a-zA-Z0-9.-]/g, '_')

		return `${prefix}/${organizationId}/${timestamp}_${random}_${safeFileName}`
	}

	// Legacy: dataset management
	async createDataset(name: string, description?: string): Promise<any> {
		this.logger.log(`Creating dataset: ${name}`)
		return { id: '1', name, description }
	}

	async getDatasets(): Promise<any[]> {
		return []
	}
}
