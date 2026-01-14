import { Injectable, OnModuleInit } from '@nestjs/common'
import { AppConfigService } from 'src/common/config/config.service'

@Injectable()
export class MediaUrlService implements OnModuleInit {
	private static responseEndpoint: string | null = null
	private static bucketName: string | null = null

	constructor(private readonly configService: AppConfigService) {}

	/**
	 * Initialize the media URL service with configuration values
	 */
	onModuleInit() {
		this.initialize()
	}

	/**
	 * Initialize the media URL service with configuration values
	 */
	private initialize() {
		if (!MediaUrlService.responseEndpoint || !MediaUrlService.bucketName) {
			MediaUrlService.responseEndpoint = this.configService.s3.responseEndpoint.replace(/\/$/, '')
			MediaUrlService.bucketName = this.configService.s3.bucket
		}
	}

	/**
	 * Build a full media URL based on key (if not already a full URL).
	 * Example: key = "images/picture_description/foo.webp" →
	 * http://itutor.sytes.net/cdn/itutor-media/images/picture_description/foo.webp
	 */
	static buildMediaUrl(key: string): string {
		if (!MediaUrlService.responseEndpoint || !MediaUrlService.bucketName) {
			throw new Error('Media URL service not initialized.')
		}

		// Skip if already a full URL
		if (/^https?:\/\//i.test(key)) {
			return key
		}

		return `${MediaUrlService.responseEndpoint}/${MediaUrlService.bucketName}/${key}`
	}

	/**
	 * Extracts the S3 object key from a full media URL.
	 * Example: https://cdn.domain.com/itutor-media/images/foo.webp →
	 *          images/foo.webp
	 */
	static extractKeyFromUrl(url: string): string | null {
		if (!MediaUrlService.responseEndpoint || !MediaUrlService.bucketName) {
			throw new Error('Media URL service not initialized.')
		}

		const base = `${MediaUrlService.responseEndpoint}/${MediaUrlService.bucketName}/`
		if (!url.startsWith(base)) return null

		return url.slice(base.length)
	}

	/**
	 * Build a public URL from an S3 key.
	 * Alias for buildMediaUrl for clarity in certain contexts.
	 */
	static buildPublicUrl(key: string): string {
		return MediaUrlService.buildMediaUrl(key)
	}
}
