import { Module } from '@nestjs/common'
import { AppConfigModule } from 'src/common/config/config.module'
import { PrismaModule } from 'src/prisma/prisma.module'
import { ExportModule } from '../export/export.module'
import { StorageModule } from '../storage/storage.module'
import { AuthModule } from '../auth/auth.module'
import { SmartCityController } from './smart-city.controller'
import { SmartCityService } from './smart-city.service'

@Module({
	imports: [PrismaModule, AuthModule, AppConfigModule, StorageModule, ExportModule],
	controllers: [SmartCityController],
	providers: [SmartCityService],
	exports: [SmartCityService],
})
export class SmartCityModule {}
