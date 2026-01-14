import { Module } from '@nestjs/common'
import { PrismaModule } from 'src/prisma/prisma.module'
import { StorageService } from './storage.service'

@Module({
	imports: [PrismaModule],
	providers: [StorageService],
	exports: [StorageService],
})
export class StorageModule {}
