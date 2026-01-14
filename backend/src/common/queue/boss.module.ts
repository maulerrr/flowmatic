import { Global, Module } from '@nestjs/common'
import { AppConfigModule } from '../config/config.module'
import { BossService } from './boss.service'

@Global()
@Module({
	imports: [AppConfigModule],
	providers: [BossService],
	exports: [BossService],
})
export class BossModule {}
