import { GetObjectCommand, HeadObjectCommand } from '@aws-sdk/client-s3'
// src/common/s3/s3.service.ts

import { Injectable, Logger, OnModuleInit } from '@nestjs/common'
import {
	S3Client,
	HeadBucketCommand,
	CreateBucketCommand,
	DeleteObjectCommand,
	S3ClientConfig,
} from '@aws-sdk/client-s3'
import { Upload } from '@aws-sdk/lib-storage'
import { AppConfigService, S3Config } from '../config/config.service'
import { Readable } from 'stream'

@Injectable()
export class S3Service implements OnModuleInit {
	/** Check if an object exists (by Key under configured bucket). */
	async exists(key: string): Promise<boolean> {
		try {
			await this.client.send(new HeadObjectCommand({ Bucket: this.cfg.bucket, Key: key }))
			return true
		} catch {
			return false
		}
	}

	/** Get object size in bytes, or null if not found. */
	async getSize(key: string): Promise<number | null> {
		try {
			const res = await this.client.send(
				new HeadObjectCommand({ Bucket: this.cfg.bucket, Key: key }),
			)
			return typeof res.ContentLength === 'number' ? res.ContentLength : null
		} catch {
			return null
		}
	}
	/** Download an object as Buffer */
	async getObjectBuffer(bucket: string, key: string): Promise<Buffer> {
		const res = await this.client.send(new GetObjectCommand({ Bucket: bucket, Key: key }))
		if (!res.Body) throw new Error('No body in S3 response')
		const stream = res.Body as Readable
		const chunks: Buffer[] = []
		for await (const chunk of stream) {
			if (typeof chunk === 'string') {
				chunks.push(Buffer.from(chunk))
			} else if (Buffer.isBuffer(chunk)) {
				chunks.push(chunk)
			} else {
				const arr = chunk as Uint8Array
				chunks.push(Buffer.from(arr))
			}
		}
		return Buffer.concat(chunks)
	}
	private readonly client: S3Client
	private readonly cfg: S3Config
	private readonly logger = new Logger(S3Service.name)

	constructor(private readonly config: AppConfigService) {
		this.cfg = this.config.s3

		const clientConfig: S3ClientConfig = {
			endpoint: this.cfg.accessEndpoint,
			region: this.cfg.region,
			credentials: {
				accessKeyId: this.cfg.accessKeyId,
				secretAccessKey: this.cfg.secretAccessKey,
			},
			forcePathStyle: Boolean(this.cfg.usePathStyle),
		}

		try {
			this.client = new S3Client(clientConfig)
		} catch (err: unknown) {
			if (err instanceof Error) {
				this.logger.error('Failed to create S3Client', err.stack)
			} else {
				this.logger.error('Failed to create S3Client')
			}
			throw err
		}
	}

	/** Called once Nest module is up */
	async onModuleInit() {
		await this.ensureBucketExists(this.cfg.bucket)
		await Promise.all([
			this.ensurePrefixObject(this.cfg.imagePrefix),
			this.ensurePrefixObject(this.cfg.audioPrefix),
			this.ensurePrefixObject('knowledge-base'),
		])
	}

	private async ensureBucketExists(bucket: string) {
		try {
			await this.client.send(new HeadBucketCommand({ Bucket: bucket }))
			this.logger.log(`Bucket "${bucket}" already exists`)
		} catch {
			this.logger.log(`Bucket "${bucket}" not found, creating…`)
			await this.client.send(new CreateBucketCommand({ Bucket: bucket }))
			this.logger.log(`Bucket "${bucket}" created`)
		}
	}

	private async ensurePrefixObject(prefix: string) {
		const key = prefix.endsWith('/') ? prefix : `${prefix}/`
		try {
			// a tiny zero-byte object, length known
			await new Upload({
				client: this.client,
				params: {
					Bucket: this.cfg.bucket,
					Key: key,
					Body: '',
				},
			}).done()
			this.logger.log(`Ensured prefix object "${key}"`)
		} catch (error: unknown) {
			if (error instanceof Error) {
				this.logger.error(`Failed to ensure prefix "${key}"`, error.stack)
			} else {
				this.logger.error(`Failed to ensure prefix "${key}"`)
			}
		}
	}

	/** Internal uploader using multipart‐capable Upload helper */
	private async uploadObject(
		key: string,
		body: Buffer | Readable,
		contentType: string,
	): Promise<string> {
		await new Upload({
			client: this.client,
			params: {
				Bucket: this.cfg.bucket,
				Key: key,
				Body: body,
				ContentType: contentType,
				ACL: 'public-read',
			},
		}).done()

		this.logger.debug(`Uploaded ${this.cfg.bucket}/${key}`)
		return key
	}

	async uploadImage(
		filename: string,
		body: Buffer | Readable,
		contentType: string,
	): Promise<string> {
		const key = `${this.cfg.imagePrefix}/${filename}`
		return this.uploadObject(key, body, contentType)
	}

	async uploadAudio(
		filename: string,
		body: Buffer | Readable,
		contentType: string,
	): Promise<string> {
		const key = `${this.cfg.audioPrefix}/${filename}`
		return this.uploadObject(key, body, contentType)
	}

	async uploadDocument(
		filename: string,
		body: Buffer | Readable,
		contentType: string,
	): Promise<string> {
		const key = `knowledge-base/${filename}`
		return this.uploadObject(key, body, contentType)
	}

	/** Generic upload with full key control - no prefix prepended */
	async upload(key: string, body: Buffer | Readable, contentType: string): Promise<string> {
		return this.uploadObject(key, body, contentType)
	}

	async delete(key: string): Promise<void> {
		await this.client.send(
			new DeleteObjectCommand({
				Bucket: this.cfg.bucket,
				Key: key,
			}),
		)
		this.logger.debug(`Deleted ${this.cfg.bucket}/${key}`)
	}

	/** Copy an object from sourceKey to destKey within the same bucket */
	async copyObject(sourceKey: string, destKey: string): Promise<void> {
		const buffer = await this.getObjectBuffer(this.cfg.bucket, sourceKey)
		// Determine content type from source extension
		let contentType = 'application/octet-stream'
		if (sourceKey.endsWith('.webp')) contentType = 'image/webp'
		else if (sourceKey.endsWith('.jpg') || sourceKey.endsWith('.jpeg')) contentType = 'image/jpeg'
		else if (sourceKey.endsWith('.png')) contentType = 'image/png'
		else if (sourceKey.endsWith('.ogg')) contentType = 'audio/ogg'
		else if (sourceKey.endsWith('.mp3')) contentType = 'audio/mpeg'
		else if (sourceKey.endsWith('.wav')) contentType = 'audio/wav'

		await this.uploadObject(destKey, buffer, contentType)
		this.logger.debug(`Copied ${sourceKey} → ${destKey}`)
	}
}
