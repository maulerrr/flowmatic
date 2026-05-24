import { Module } from '@nestjs/common'
import { StorageService } from './storage.service'
import { AppConfigModule } from 'src/common/config/config.module'

@Module({
	imports: [AppConfigModule],
	providers: [StorageService],
	exports: [StorageService],
})
export class StorageModule {}
