import { Module } from '@nestjs/common'
import { QualityController } from './quality.controller'
import { QualityService } from './quality.service'
import { AuthModule } from '../auth/auth.module'

@Module({
	imports: [AuthModule],
	controllers: [QualityController],
	providers: [QualityService],
	exports: [QualityService],
})
export class QualityModule {}
