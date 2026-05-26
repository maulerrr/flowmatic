import { Module } from '@nestjs/common'
import { AppConfigModule } from 'src/common/config/config.module'
import { AuthModule } from '../auth/auth.module'
import { ExportModule } from '../export/export.module'
import { HuggingFaceIntegrationService } from './huggingface.integration.service'
import { IntegrationsController } from './integrations.controller'

@Module({
	imports: [AuthModule, ExportModule, AppConfigModule],
	controllers: [IntegrationsController],
	providers: [HuggingFaceIntegrationService],
	exports: [HuggingFaceIntegrationService],
})
export class IntegrationsModule {}
