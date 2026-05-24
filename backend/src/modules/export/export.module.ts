import { Module } from '@nestjs/common'
import { ExportController } from './export.controller'
import { ExportService } from './export.service'
import { ExportAdapterRegistry } from './adapters/registry'
import { StorageModule } from '../storage/storage.module'
import { AuthModule } from '../auth/auth.module'
import { PrismaModule } from 'src/prisma/prisma.module'
import { AppConfigModule } from 'src/common/config/config.module'

@Module({
	imports: [StorageModule, AuthModule, PrismaModule, AppConfigModule],
	controllers: [ExportController],
	providers: [ExportAdapterRegistry, ExportService],
	exports: [ExportService, ExportAdapterRegistry],
})
export class ExportModule {}
