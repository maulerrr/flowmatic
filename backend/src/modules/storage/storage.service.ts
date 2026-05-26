import { Injectable, Logger } from '@nestjs/common'
import {
	S3Client,
	S3ClientConfig,
	CreateBucketCommand,
	HeadBucketCommand,
	ListObjectsV2Command,
	PutObjectCommand,
	GetObjectCommand,
	DeleteObjectCommand,
} from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { Readable } from 'stream'
import { AppConfigService } from 'src/common/config/config.service'

export interface S3UploadOptions {
	bucket: string
	key: string
	body: Buffer | string
	contentType?: string
	metadata?: Record<string, string>
}

export interface S3ConnectionOptions {
	endpoint?: string | null
	region?: string | null
	accessKeyId: string
	secretAccessKey: string
	usePathStyle?: boolean
}

export interface S3ListedObject {
	key: string
	size: number
	lastModified: Date | null
	etag: string | null
}

@Injectable()
export class StorageService {
	private readonly logger = new Logger(StorageService.name)
	private s3Client: S3Client

	constructor(private readonly config: AppConfigService) {
		const { accessKeyId, secretAccessKey, accessEndpoint, region, usePathStyle } = this.config.s3

		if (accessKeyId && secretAccessKey) {
			const clientConfig: S3ClientConfig = {
				region,
				credentials: {
					accessKeyId,
					secretAccessKey,
				},
			}

			if (accessEndpoint) {
				clientConfig.endpoint = accessEndpoint
				clientConfig.forcePathStyle = usePathStyle
			}

			this.s3Client = new S3Client(clientConfig)
		}
	}

	get defaultBucket(): string {
		return this.config.s3.bucket
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

	async getSignedDownloadUrl(
		bucket: string,
		key: string,
		expiresIn: number = 3600,
	): Promise<string> {
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

		const s3Bucket = bucket || this.defaultBucket

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
				const stream = response.Body as Readable
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

	createClient(options?: S3ConnectionOptions): S3Client {
		if (!options) {
			if (!this.s3Client) {
				throw new Error('S3 credentials not configured')
			}
			return this.s3Client
		}

		const clientConfig: S3ClientConfig = {
			region: options.region || this.config.s3.region,
			credentials: {
				accessKeyId: options.accessKeyId,
				secretAccessKey: options.secretAccessKey,
			},
		}

		if (options.endpoint) {
			clientConfig.endpoint = options.endpoint
			clientConfig.forcePathStyle = options.usePathStyle ?? this.config.s3.usePathStyle
		}

		return new S3Client(clientConfig)
	}

	async ensureBucketExists(
		bucket: string,
		options?: S3ConnectionOptions,
	): Promise<void> {
		const client = this.createClient(options)
		try {
			await client.send(new HeadBucketCommand({ Bucket: bucket }))
			return
		} catch (error) {
			this.logger.warn(`Bucket ${bucket} not found or inaccessible, attempting to create it`)
		}

		await client.send(new CreateBucketCommand({ Bucket: bucket }))
	}

	async uploadObject(
		options: S3UploadOptions,
		connection?: S3ConnectionOptions,
	): Promise<string> {
		const client = this.createClient(connection)
		await this.ensureBucketExists(options.bucket, connection)

		try {
			const command = new PutObjectCommand({
				Bucket: options.bucket,
				Key: options.key,
				Body: options.body,
				ContentType: options.contentType || 'application/octet-stream',
				Metadata: options.metadata,
			})

			await client.send(command)
			this.logger.debug(`File uploaded to S3: s3://${options.bucket}/${options.key}`)
			return options.key
		} catch (error) {
			this.logger.error(`Failed to upload file to S3: ${error}`)
			throw error
		}
	}

	async listObjects(
		bucket: string,
		options?: S3ConnectionOptions,
		prefix?: string,
		maxKeys: number = 100,
	): Promise<S3ListedObject[]> {
		const client = this.createClient(options)
		const response = await client.send(
			new ListObjectsV2Command({
				Bucket: bucket,
				Prefix: prefix || undefined,
				MaxKeys: maxKeys,
			}),
		)
		return (response.Contents ?? []).map(item => ({
			key: item.Key ?? '',
			size: item.Size ?? 0,
			lastModified: item.LastModified ?? null,
			etag: item.ETag ?? null,
		}))
	}

	async getObjectBody(
		bucket: string,
		key: string,
		connection?: S3ConnectionOptions,
	): Promise<Buffer | null> {
		const client = this.createClient(connection)
		try {
			const response = await client.send(
				new GetObjectCommand({
					Bucket: bucket,
					Key: key,
				}),
			)
			const bytes = await response.Body?.transformToByteArray()
			return bytes ? Buffer.from(bytes) : Buffer.alloc(0)
		} catch (error) {
			const code =
				error && typeof error === 'object' && 'name' in error
					? String((error as { name?: string }).name)
					: ''
			if (code === 'NoSuchKey' || code === 'NotFound') return null
			throw error
		}
	}

	async appendNdjsonLine(
		options: S3UploadOptions,
		connection?: S3ConnectionOptions,
	): Promise<string> {
		const existing = await this.getObjectBody(options.bucket, options.key, connection)
		const line = typeof options.body === 'string' ? options.body : options.body.toString('utf8')
		const body =
			existing && existing.length > 0 ? `${existing.toString('utf8').replace(/\n$/, '')}\n${line}` : line
		return this.uploadObject(
			{
				...options,
				body,
				contentType: options.contentType ?? 'application/x-ndjson',
			},
			connection,
		)
	}
}
