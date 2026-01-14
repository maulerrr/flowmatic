import { Injectable, Logger, OnModuleDestroy, OnModuleInit } from '@nestjs/common'
// Use namespace import to avoid any potential type ambiguity with named imports
import * as amqp from 'amqplib'
import { v4 as uuid } from 'uuid'
import { QueueClient, QueueMessage, QueuePublishOptions } from '../queue.tokens'
import { AppConfigService } from '../../config/config.service'

@Injectable()
export class RabbitMQQueueAdapter implements QueueClient, OnModuleInit, OnModuleDestroy {
	private readonly logger = new Logger(RabbitMQQueueAdapter.name)
	// Narrowed shape we actually use to avoid type resolution issues some configs show with amqplib Connection
	private connection?: { createChannel: () => Promise<amqp.Channel>; close: () => Promise<void> }
	private channel?: amqp.Channel

	private parseMessage<T>(buffer: Buffer): QueueMessage<T> {
		let parsed: unknown
		try {
			parsed = JSON.parse(buffer.toString('utf8'))
		} catch (e) {
			throw new Error(`RabbitMQQueueAdapter: invalid JSON payload (${(e as Error).message})`)
		}
		if (!parsed || typeof parsed !== 'object') {
			throw new Error('RabbitMQQueueAdapter: parsed payload not an object')
		}
		const obj = parsed as Partial<QueueMessage<T>>
		if (typeof obj.type !== 'string') throw new Error('RabbitMQQueueAdapter: message.type missing')
		if (!('payload' in obj)) throw new Error('RabbitMQQueueAdapter: message.payload missing')
		return {
			id: typeof obj.id === 'string' ? obj.id : undefined,
			type: obj.type,
			payload: obj.payload as T,
			occurredAt: typeof obj.occurredAt === 'string' ? obj.occurredAt : new Date().toISOString(),
			version: typeof obj.version === 'number' ? obj.version : 1,
			dedupeKey: typeof obj.dedupeKey === 'string' ? obj.dedupeKey : undefined,
		}
	}

	constructor(private readonly config: AppConfigService) {}

	async onModuleInit(): Promise<void> {
		const url: string | undefined = this.config.queue.rabbitUrl || process.env.RABBITMQ_URL
		if (!url) {
			this.logger.warn('RabbitMQ URL not provided; adapter disabled')
			return
		}
		try {
			const raw = (await amqp.connect(url)) as unknown as {
				createChannel: () => Promise<amqp.Channel>
				close: () => Promise<void>
			}
			this.connection = raw
			this.channel = await raw.createChannel()
			this.logger.log('RabbitMQ connected')
		} catch (e) {
			this.logger.error(`RabbitMQ connection failed: ${(e as Error).message}`)
		}
	}

	async onModuleDestroy(): Promise<void> {
		await this.close()
	}

	private ensureChannel(): amqp.Channel {
		const ch = this.channel
		if (!ch) throw new Error('RabbitMQ channel not initialized')
		return ch
	}

	async publish<T>(
		queue: string,
		message: QueueMessage<T>,
		options?: QueuePublishOptions,
	): Promise<void> {
		const ch = this.ensureChannel()
		await ch.assertQueue(queue, { durable: true })
		const enriched: QueueMessage<T> = {
			...message,
			id: message.id ?? uuid(),
			occurredAt: message.occurredAt || new Date().toISOString(),
		}
		const ttl = options?.expiresInMs ?? this.config.queue.defaultTtlMs
		ch.sendToQueue(queue, Buffer.from(JSON.stringify(enriched)), {
			persistent: true,
			expiration: ttl && ttl > 0 ? String(ttl) : undefined,
		})
	}

	async subscribe<T>(
		queue: string,
		handler: (msg: QueueMessage<T>) => Promise<void>,
	): Promise<void> {
		const ch = this.ensureChannel()
		await ch.assertQueue(queue, { durable: true })
		await ch.consume(queue, (msg: amqp.ConsumeMessage | null) => {
			if (!msg) return
			void (async () => {
				try {
					const parsed = this.parseMessage<T>(msg.content)
					await handler(parsed)
					ch.ack(msg)
				} catch (e) {
					this.logger.error(`Handler error for queue ${queue}: ${(e as Error).message}`)
					ch.nack(msg, false, false)
				}
			})()
		})
	}

	async close(): Promise<void> {
		try {
			if (this.channel) {
				await this.channel.close()
				this.channel = undefined
			}
			if (this.connection) {
				await this.connection.close()
				this.connection = undefined
			}
		} catch (e) {
			this.logger.warn(`Error closing RabbitMQ: ${(e as Error).message}`)
		}
	}
}
