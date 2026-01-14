import { Module } from '@nestjs/common'
import { PrismaModule } from 'src/prisma/prisma.module'
import { StorageModule } from '../storage/storage.module'
import { AuthModule } from '../auth/auth.module'
import { PipelineModule } from '../pipeline/pipeline.module'
import { IngestionController } from './ingestion.controller'
import { IngestionService } from './ingestion.service'

@Module({
	imports: [PrismaModule, StorageModule, AuthModule, PipelineModule],
	controllers: [IngestionController],
	providers: [IngestionService],
	exports: [IngestionService],
})
export class IngestionModule {}
