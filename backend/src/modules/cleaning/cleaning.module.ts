import { Module } from '@nestjs/common'
import { CleaningController } from './cleaning.controller'
import { CleaningService } from './cleaning.service'
import { AuthModule } from '../auth/auth.module'

@Module({
	imports: [AuthModule],
	controllers: [CleaningController],
	providers: [CleaningService],
	exports: [CleaningService],
})
export class CleaningModule {}
