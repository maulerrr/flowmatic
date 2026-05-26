import { Module } from '@nestjs/common'
import { AppConfigModule } from 'src/common/config/config.module'
import { PrismaModule } from 'src/prisma/prisma.module'
import { ExportModule } from '../export/export.module'
import { IntegrationsModule } from '../integrations/integrations.module'
import { StorageModule } from '../storage/storage.module'
import { AuthModule } from '../auth/auth.module'
import { QualityModule } from '../quality/quality.module'
import { SmartCityController } from './smart-city.controller'
import { SmartCityService } from './smart-city.service'
import { PipelineCopilotService } from './pipeline-copilot.service'
import { PipelineDataProfilerService } from './pipeline-data-profiler.service'
import { PipelineAnalysisEngineService } from './pipeline-analysis-engine.service'
import { PipelineInsightPlannerService } from './pipeline-insight-planner.service'
import { PipelineInsightVizBuilderService } from './pipeline-insight-viz-builder.service'
import { PipelineInsightService } from './pipeline-insight.service'
import { PipelineInsightScheduler } from './pipeline-insight.scheduler'
import { PipelineModelRegistryService } from './pipeline-model-registry.service'
import { PipelineModelRouterService } from './pipeline-model-router.service'
import { PipelineAutoRoutingService } from './pipeline-auto-routing.service'

@Module({
	imports: [PrismaModule, AuthModule, AppConfigModule, StorageModule, ExportModule, IntegrationsModule, QualityModule],
	controllers: [SmartCityController],
	providers: [
		SmartCityService,
		PipelineCopilotService,
		PipelineDataProfilerService,
		PipelineAnalysisEngineService,
		PipelineInsightPlannerService,
		PipelineInsightVizBuilderService,
		PipelineInsightService,
		PipelineInsightScheduler,
		PipelineModelRegistryService,
		PipelineModelRouterService,
		PipelineAutoRoutingService,
	],
	exports: [SmartCityService, PipelineInsightService],
})
export class SmartCityModule {}
