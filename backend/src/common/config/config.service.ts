import { Injectable } from '@nestjs/common'
import { ConfigService as NestConfigService } from '@nestjs/config'

export interface DatabaseConfig {
	url: string
}
export interface JwtConfig {
	secret: string
	expiresIn: string
}
export interface ClientConfig {
	url: string
	cookieDomain: string
}
export interface WhatsappConfig {
	version: string
	phoneId: string
	accessToken: string
}
export interface OpenAIConfig {
	apiKey: string
	model: string // legacy single model (kept for backward compatibility)
}

export interface SecurityConfig {
	backendCorsOrigins: string[]
	allowedHosts: string[]
	exportCredentialsSecret: string
}

export interface S3Config {
	accessEndpoint: string
	responseEndpoint: string
	region: string
	accessKeyId: string
	secretAccessKey: string
	bucket: string
	imagePrefix: string // e.g. “images/”
	audioPrefix: string // e.g. “audio/”
	usePathStyle: boolean
}

export interface QueueConfig {
	driver: string
	rabbitUrl?: string
	defaultTtlMs?: number
}

export interface ServerConfig {
	port: number
	requestTimeoutMs: number
}

export interface QdrantConfig {
	url: string
	apiKey?: string
	preferGrpc?: boolean
}

export interface EmbeddingConfig {
	endpointUrl: string
	apiKey?: string
}

export interface SensorSimulatorConfig {
	baseUrl: string
}

export interface ModelInferenceConfig {
	baseUrl: string
}

export interface HuggingFaceConfig {
	token?: string
}

@Injectable()
export class AppConfigService {
	constructor(private readonly config: NestConfigService) {}

	private get<T = string>(key: string, defaultValue?: T): T {
		return defaultValue !== undefined
			? this.config.get<T>(key, defaultValue)
			: this.config.get<T>(key)!
	}

	get database(): DatabaseConfig {
		return {
			url: this.get<string>('DATABASE_URL'),
		}
	}

	get client(): ClientConfig {
		const url = this.get<string>('CLIENT_URL', 'http://localhost:3000')
		return {
			url,
			cookieDomain: new URL(url).hostname,
		}
	}

	get whatsapp(): WhatsappConfig {
		return {
			version: this.get<string>('WHATSAPP_API_VERSION'),
			phoneId: this.get<string>('WHATSAPP_PHONE_NUMBER_ID'),
			accessToken: this.get<string>('WHATSAPP_ACCESS_TOKEN'),
		}
	}

	get openai(): OpenAIConfig {
		return {
			apiKey: this.get<string>('OPENAI_API_KEY'),
			model: this.get<string>('OPENAI_MODEL', 'gpt-4.1-2025-04-14'),
		}
	}

	get s3(): S3Config {
		return {
			accessEndpoint: this.get<string>('S3_ACCESS_ENDPOINT', 'http://localhost:9000'),
			responseEndpoint: this.get<string>('S3_RESPONSE_ENDPOINT', 'http://localhost:9000'),
			region: this.get<string>('S3_REGION', 'ap-northeast-2'),
			accessKeyId: this.get<string>('S3_ACCESS_KEY', 'minio'),
			secretAccessKey: this.get<string>('S3_SECRET_KEY', 'minio123'),
			bucket: this.get<string>('S3_BUCKET', 'flowmatic-media'),
			imagePrefix: this.get<string>('S3_IMAGE_PREFIX', 'images'),
			audioPrefix: this.get<string>('S3_AUDIO_PREFIX', 'audio'),
			usePathStyle: this.get<boolean>('S3_PATH_STYLE', true),
		}
	}

	get nodeEnv(): string {
		return this.get<string>('NODE_ENV', 'development')
	}

	get server(): ServerConfig {
		return {
			port: Number(this.get<string>('SERVER_PORT', '8080')),
			requestTimeoutMs: Number(this.get<string>('REQUEST_TIMEOUT_MS', '300000')),
		}
	}

	get isProduction(): boolean {
		return this.nodeEnv === 'production'
	}

	get security(): SecurityConfig {
		return {
			backendCorsOrigins: this.get<string>('SECURITY_BACKEND_CORS_ORIGINS', '').split(','),
			allowedHosts: this.get<string>('SECURITY_ALLOWED_HOSTS', '').split(','),
			exportCredentialsSecret: this.get<string>(
				'EXPORT_CREDENTIALS_SECRET',
				this.get<string>('JWT_SECRET', this.database.url),
			),
		}
	}

	get queue(): QueueConfig {
		return {
			driver: this.get<string>('QUEUE_DRIVER', 'pgboss'),
			rabbitUrl: this.get<string>('RABBITMQ_URL', ''),
			defaultTtlMs: this.get<number>('QUEUE_DEFAULT_TTL_MS', 0),
		}
	}

	get qdrant(): QdrantConfig {
		return {
			url: this.get<string>('QDRANT_URL', 'http://localhost:6333'),
			apiKey: this.get<string | undefined>('QDRANT_API_KEY'),
			preferGrpc: this.get<string>('QDRANT_PREFER_GRPC', 'false') === 'true',
		}
	}

	get embedding(): EmbeddingConfig {
		return {
			endpointUrl: this.get<string>('EMBEDDING_ENDPOINT_URL', 'http://localhost:8000'),
			apiKey: this.get<string | undefined>('EMBEDDING_API_KEY'),
		}
	}

	get sensorSimulator(): SensorSimulatorConfig {
		return {
			baseUrl: this.get<string>('SENSOR_SIMULATOR_URL', 'http://localhost:8091'),
		}
	}

	get modelInference(): ModelInferenceConfig {
		return {
			baseUrl: this.get<string>('MODEL_INFERENCE_URL', 'http://localhost:8093'),
		}
	}

	get huggingFace(): HuggingFaceConfig {
		const token =
			this.get<string | undefined>('HUGGINGFACE_TOKEN') ??
			this.get<string | undefined>('HF_TOKEN')
		return {
			token: token?.trim() || undefined,
		}
	}
}
