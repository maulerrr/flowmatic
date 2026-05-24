import { Module } from '@nestjs/common'
import { PrismaModule } from 'src/prisma/prisma.module'
import { AppConfigModule } from 'src/common/config/config.module'
import { AuthController } from './auth.controller'
import { AuthContextService } from './auth-context.service'
import { AuthGuard } from './auth.guard'

@Module({
	imports: [PrismaModule, AppConfigModule],
	controllers: [AuthController],
	providers: [AuthContextService, AuthGuard],
	exports: [AuthContextService, AuthGuard],
})
export class AuthModule {}
