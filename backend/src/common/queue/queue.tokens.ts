export const QUEUE_CLIENT = Symbol('QUEUE_CLIENT')

export enum QueueDriver {
	PG_BOSS = 'pgboss',
	RABBITMQ = 'rabbitmq',
}

export interface QueuePublishOptions {
	delayMs?: number
	expiresInMs?: number
}

export type QueueMessage<T = unknown> = {
	id?: string
	type: string
	payload: T
	occurredAt: string
	version?: number
	dedupeKey?: string
}

export interface QueueClient {
	publish<T = unknown>(
		queue: string,
		message: QueueMessage<T>,
		options?: QueuePublishOptions,
	): Promise<void>
	subscribe<T = unknown>(
		queue: string,
		handler: (msg: QueueMessage<T>) => Promise<void>,
	): Promise<void>
	close(): Promise<void>
}
