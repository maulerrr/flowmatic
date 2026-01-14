import { Injectable, Logger, OnModuleDestroy, OnModuleInit } from '@nestjs/common'
import PgBoss from 'pg-boss'
import { AppConfigService } from '../config/config.service'

type PublishOptions = Parameters<PgBoss['publish']>[2]

@Injectable()
export class BossService implements OnModuleInit, OnModuleDestroy {
	private readonly logger = new Logger(BossService.name)
	private boss?: PgBoss

	constructor(private readonly config: AppConfigService) {}

	async onModuleInit() {
		const connectionString = this.config.database.url
		if (!connectionString) {
			this.logger.warn('DATABASE_URL not set; pg-boss will not start')
			return
		}
		this.boss = new PgBoss({ connectionString })
		await this.boss.start()
		this.logger.log('pg-boss started')
	}

	async onModuleDestroy() {
		if (this.boss) {
			await this.boss.stop()
			this.logger.log('pg-boss stopped')
		}
	}

	get instance(): PgBoss | undefined {
		return this.boss
	}

	async publish(queue: string, data?: object, options?: PublishOptions): Promise<void> {
		if (!this.boss) throw new Error('pg-boss not started')
		const b = this.boss as unknown as { publish: (q: string, d?: any, o?: any) => Promise<unknown> }
		await b.publish(queue, data, options)
	}

	async subscribe(queue: string, handler: (job: unknown) => Promise<void> | void): Promise<void> {
		if (!this.boss) throw new Error('pg-boss not started')
		const b = this.boss as unknown as {
			subscribe: (q: string, h: (job: any) => any) => Promise<unknown>
		}
		await b.subscribe(queue, handler as (job: any) => any)
	}

	async subscribeWithOptions(
		queue: string,
		options: any,
		handler: (job: unknown) => Promise<void> | void,
	): Promise<void> {
		if (!this.boss) throw new Error('pg-boss not started')
		const b = this.boss as unknown as {
			subscribe: (q: string, o: any, h: (job: any) => any) => Promise<unknown>
		}
		await b.subscribe(queue, options, handler as (job: any) => any)
	}
}
