import { Module } from '@nestjs/common'
import { PrismaModule } from 'src/prisma/prisma.module'
import { QualityModule } from '../quality/quality.module'
import { CleaningModule } from '../cleaning/cleaning.module'
import { StorageModule } from '../storage/storage.module'
import { AuthModule } from '../auth/auth.module'
import { QueueModule } from 'src/common/queue/queue.module'
import { AppConfigModule } from 'src/common/config/config.module'
import { PipelineController } from './pipeline.controller'
import { PipelineService } from './pipeline.service'

@Module({
	imports: [
		PrismaModule,
		QualityModule,
		CleaningModule,
		StorageModule,
		AuthModule,
		QueueModule.register(),
		AppConfigModule,
	],
	controllers: [PipelineController],
	providers: [PipelineService],
	exports: [PipelineService],
})
export class PipelineModule {}
