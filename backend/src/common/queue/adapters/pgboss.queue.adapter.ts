import { Injectable, Logger, OnModuleDestroy, OnModuleInit, Optional } from '@nestjs/common'
import { v4 as uuid } from 'uuid'
import { BossService } from '../boss.service'
import { QueueClient, QueueMessage, QueuePublishOptions } from '../queue.tokens'

interface PgBossJob<T> {
	data: T
}

@Injectable()
export class PgBossQueueAdapter implements QueueClient, OnModuleInit, OnModuleDestroy {
	private readonly logger = new Logger(PgBossQueueAdapter.name)

	constructor(@Optional() private readonly bossService?: BossService) {}

	async onModuleInit(): Promise<void> {
		await Promise.resolve()
		if (!this.bossService?.instance) {
			this.logger.warn('PgBossQueueAdapter: Boss service not started (DATABASE_URL missing?)')
		}
	}

	async onModuleDestroy(): Promise<void> {
		// underlying BossService handles stop
	}

	async publish<T>(
		queue: string,
		message: QueueMessage<T>,
		options?: QueuePublishOptions,
	): Promise<void> {
		if (!this.bossService?.instance) throw new Error('PgBossQueueAdapter: boss not started')
		const payload = { ...message, id: message.id ?? uuid() }
		await this.bossService.publish(queue, payload, {
			startAfter: options?.delayMs ? new Date(Date.now() + options.delayMs) : undefined,
		})
	}

	async subscribe<T>(
		queue: string,
		handler: (msg: QueueMessage<T>) => Promise<void>,
	): Promise<void> {
		if (!this.bossService?.instance) throw new Error('PgBossQueueAdapter: boss not started')
		await this.bossService.subscribe(queue, async (job: unknown) => {
			if (!job || typeof job !== 'object' || !('data' in job)) {
				throw new Error('PgBossQueueAdapter: job missing data property')
			}
			const { data } = job as PgBossJob<unknown>
			if (
				!data ||
				typeof data !== 'object' ||
				typeof (data as Record<string, unknown>).type !== 'string'
			) {
				throw new Error('PgBossQueueAdapter: invalid message shape')
			}
			await handler(data as QueueMessage<T>)
		})
	}

	async close(): Promise<void> {
		// pg-boss closed by BossService lifecycle
	}
}
