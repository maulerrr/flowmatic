import { Module } from '@nestjs/common'
import { DevtoolsModule } from '@nestjs/devtools-integration'
import { LoggerModule } from 'nestjs-pino'
import { ScheduleModule } from '@nestjs/schedule'

import { AppConfigModule } from './common/config/config.module'
import { BossModule } from './common/queue/boss.module'
import { PrismaModule } from './prisma/prisma.module'

// Data Preparation Modules
import { IngestionModule } from './modules/ingestion/ingestion.module'
import { QualityModule } from './modules/quality/quality.module'
import { CleaningModule } from './modules/cleaning/cleaning.module'
import { PipelineModule } from './modules/pipeline/pipeline.module'
import { ExportModule } from './modules/export/export.module'
import { StorageModule } from './modules/storage/storage.module'
import { AuthModule } from './modules/auth/auth.module'
import { SmartCityModule } from './modules/smart-city/smart-city.module'

@Module({
	imports: [
		// Dev / logging
		DevtoolsModule.register({ http: process.env.NODE_ENV !== 'production' }),
		LoggerModule.forRoot({
			pinoHttp: {
				level: process.env.NODE_ENV !== 'production' ? 'debug' : 'info',
				transport: process.env.NODE_ENV !== 'production' ? { target: 'pino-pretty' } : undefined,
			},
		}),

		// Global config & DB
		AppConfigModule,
		ScheduleModule.forRoot(),
		BossModule,
		PrismaModule,

		// Auth & Data Preparation Pipeline
		AuthModule,
		IngestionModule,
		QualityModule,
		CleaningModule,
		PipelineModule,
		ExportModule,
		StorageModule,
		SmartCityModule,
	],
})
export class AppModule {}
