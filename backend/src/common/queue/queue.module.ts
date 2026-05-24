import { DynamicModule, Global, Module } from '@nestjs/common'
import { AppConfigModule } from '../config/config.module'
import { AppConfigService } from '../config/config.service'
import { QUEUE_CLIENT, QueueClient, QueueDriver } from './queue.tokens'
import { RabbitMQQueueAdapter } from './adapters/rabbitmq.queue.adapter'
import { PgBossQueueAdapter } from './adapters/pgboss.queue.adapter'
import { BossModule } from './boss.module'

@Global()
@Module({})
export class QueueModule {
	static register(): DynamicModule {
		return {
			module: QueueModule,
			imports: [AppConfigModule, BossModule],
			providers: [
				RabbitMQQueueAdapter,
				PgBossQueueAdapter,
				{
					provide: QUEUE_CLIENT,
					inject: [AppConfigService, RabbitMQQueueAdapter, PgBossQueueAdapter],
					useFactory: (
						config: AppConfigService,
						rabbit: RabbitMQQueueAdapter,
						pgboss: PgBossQueueAdapter,
					): QueueClient => {
						const driver = config.queue.driver as QueueDriver
						return driver === QueueDriver.RABBITMQ ? rabbit : pgboss
					},
				},
			],
			exports: [QUEUE_CLIENT],
		}
	}
}
