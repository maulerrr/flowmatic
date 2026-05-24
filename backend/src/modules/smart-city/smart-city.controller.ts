import { Body, Controller, Delete, Get, Param, Patch, Post, Query, Req, UseGuards } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { AuthGuard } from '../auth/auth.guard'
import { AggregateFederatedRoundDto } from './dto/aggregate-federated-round.dto'
import { BackfillPipelineDto } from './dto/backfill-pipeline.dto'
import { ConnectFederatedDto } from './dto/connect-federated.dto'
import { CreateDataLakeDto } from './dto/create-data-lake.dto'
import { CreateExportTargetDto } from './dto/create-export-target.dto'
import { ExportPipelineStageDto } from './dto/export-pipeline-stage.dto'
import { StartFederatedRoundDto } from './dto/start-federated-round.dto'
import { CreateSensorSourceDto } from './dto/create-sensor-source.dto'
import { CreateSmartCityPipelineDto } from './dto/create-smart-city-pipeline.dto'
import { SubmitFederatedUpdateDto } from './dto/submit-federated-update.dto'
import { SyncFederatedGlobalModelDto } from './dto/sync-federated-global-model.dto'
import { TestProcessingDto } from './dto/test-processing.dto'
import { UpdateDataLakeDto } from './dto/update-data-lake.dto'
import { UpdateExportTargetDto } from './dto/update-export-target.dto'
import { UpdatePipelineGraphDto } from './dto/update-pipeline-graph.dto'
import { UpdateSensorSourceDto } from './dto/update-sensor-source.dto'
import { UpdateSmartCityPipelineDto } from './dto/update-smart-city-pipeline.dto'
import { TrainModelDto } from './dto/train-model.dto'
import { SmartCityService } from './smart-city.service'

@ApiTags('smart-city')
@Controller('smart-city')
@UseGuards(AuthGuard)
export class SmartCityController {
	constructor(private readonly smartCity: SmartCityService) {}

	@Get('pipelines')
	async listPipelines(@Req() req: AuthenticatedRequest) {
		return { success: true, data: await this.smartCity.listPipelines(req.authContext!) }
	}

	@Get('simulator/presets')
	async getSimulatorPresets() {
		return { success: true, data: await this.smartCity.getSimulatorPresets() }
	}

	@Post('pipelines')
	async createPipeline(@Req() req: AuthenticatedRequest, @Body() body: CreateSmartCityPipelineDto) {
		return { success: true, data: await this.smartCity.createPipeline(req.authContext!, body) }
	}

	@Get('pipelines/:id')
	async getPipeline(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.getPipeline(req.authContext!, id) }
	}

	@Patch('pipelines/:id')
	async updatePipeline(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: UpdateSmartCityPipelineDto,
	) {
		return { success: true, data: await this.smartCity.updatePipeline(req.authContext!, id, body) }
	}

	@Delete('pipelines/:id')
	async deletePipeline(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.deletePipeline(req.authContext!, id) }
	}

	@Patch('pipelines/:id/graph')
	async updateGraph(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: UpdatePipelineGraphDto,
	) {
		return { success: true, data: await this.smartCity.updateGraph(req.authContext!, id, body.graph) }
	}

	@Get('pipelines/:id/dashboard')
	async getDashboard(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.getDashboard(req.authContext!, id) }
	}

	@Get('pipelines/:id/observability')
	async getObservability(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.getObservability(req.authContext!, id) }
	}

	@Get('pipelines/:id/n8n-workflow')
	async getN8nWorkflow(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.getN8nWorkflow(req.authContext!, id) }
	}

	@Post('pipelines/:id/exports')
	async exportPipelineStage(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: ExportPipelineStageDto,
	) {
		return { success: true, data: await this.smartCity.exportPipelineStage(req.authContext!, id, body) }
	}

	@Post('pipelines/:id/backfill')
	async backfillPipeline(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: BackfillPipelineDto,
	) {
		return { success: true, data: await this.smartCity.backfillPipeline(req.authContext!, id, body) }
	}

	@Get('pipelines/:id/export-targets')
	async listExportTargets(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.listExportTargets(req.authContext!, id) }
	}

	@Get('pipelines/:id/export-runs')
	async listExportRuns(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.listExportRuns(req.authContext!, id) }
	}

	@Post('pipelines/:id/export-targets')
	async createExportTarget(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: CreateExportTargetDto,
	) {
		return { success: true, data: await this.smartCity.createExportTarget(req.authContext!, id, body) }
	}

	@Patch('export-targets/:id')
	async updateExportTarget(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: UpdateExportTargetDto,
	) {
		return { success: true, data: await this.smartCity.updateExportTarget(req.authContext!, id, body) }
	}

	@Post('export-targets/:id/run')
	async runExportTarget(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.runExportTarget(req.authContext!, id) }
	}

	@Post('pipelines/:id/federated/connect')
	async connectFederated(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: ConnectFederatedDto,
	) {
		return { success: true, data: await this.smartCity.connectFederated(req.authContext!, id, body) }
	}

	@Post('pipelines/:id/federated/test')
	async testFederated(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.testFederated(req.authContext!, id) }
	}

	@Post('pipelines/:id/federated/disconnect')
	async disconnectFederated(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.disconnectFederated(req.authContext!, id) }
	}

	@Get('pipelines/:id/federated/rounds')
	async listFederatedRounds(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.listFederatedRounds(req.authContext!, id) }
	}

	@Post('pipelines/:id/federated/rounds')
	async startFederatedRound(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: StartFederatedRoundDto,
	) {
		return { success: true, data: await this.smartCity.startFederatedRound(req.authContext!, id, body) }
	}

	@Post('pipelines/:id/federated/rounds/:roundId/submit')
	async submitFederatedUpdate(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Param('roundId') roundId: string,
		@Body() body: SubmitFederatedUpdateDto,
	) {
		return {
			success: true,
			data: await this.smartCity.submitFederatedUpdate(req.authContext!, id, roundId, body),
		}
	}

	@Post('pipelines/:id/federated/rounds/:roundId/aggregate')
	async aggregateFederatedRound(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Param('roundId') roundId: string,
		@Body() body: AggregateFederatedRoundDto,
	) {
		return {
			success: true,
			data: await this.smartCity.aggregateFederatedRound(req.authContext!, id, roundId, body),
		}
	}

	@Post('pipelines/:id/federated/sync-global-model')
	async syncFederatedGlobalModel(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: SyncFederatedGlobalModelDto,
	) {
		return {
			success: true,
			data: await this.smartCity.syncFederatedGlobalModel(req.authContext!, id, body),
		}
	}

	@Get('pipelines/:id/sources')
	async listSources(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.listSources(req.authContext!, id) }
	}

	@Post('pipelines/:id/sources')
	async createSource(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: CreateSensorSourceDto,
	) {
		return { success: true, data: await this.smartCity.createSource(req.authContext!, id, body) }
	}

	@Get('pipelines/:id/events')
	async listEvents(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Query('limit') limit?: string,
	) {
		return {
			success: true,
			data: await this.smartCity.listEvents(req.authContext!, id, limit ? Number(limit) : 50),
		}
	}

	@Get('pipelines/:id/http-poll')
	async pollHttpStream(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.pollHttpStream(req.authContext!, id) }
	}

	@Patch('sources/:id')
	async updateSource(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: UpdateSensorSourceDto,
	) {
		return { success: true, data: await this.smartCity.updateSource(req.authContext!, id, body) }
	}

	@Delete('sources/:id')
	async deleteSource(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.deleteSource(req.authContext!, id) }
	}

	@Post('sources/:id/start')
	async startSource(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.startSource(req.authContext!, id) }
	}

	@Post('sources/:id/stop')
	async stopSource(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.stopSource(req.authContext!, id) }
	}

	@Post('sources/:id/test')
	async testSource(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.testSource(req.authContext!, id) }
	}

	@Get('data-lakes')
	async listDataLakes(@Req() req: AuthenticatedRequest) {
		return { success: true, data: await this.smartCity.listDataLakes(req.authContext!) }
	}

	@Get('pipelines/:id/data-lake-objects')
	async listDataLakeObjects(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Query('stage') stage?: string,
	) {
		return {
			success: true,
			data: await this.smartCity.listDataLakeObjects(req.authContext!, id, stage),
		}
	}

	@Post('data-lakes')
	async createDataLake(@Req() req: AuthenticatedRequest, @Body() body: CreateDataLakeDto) {
		return { success: true, data: await this.smartCity.createDataLake(req.authContext!, body) }
	}

	@Patch('data-lakes/:id')
	async updateDataLake(
		@Req() req: AuthenticatedRequest,
		@Param('id') id: string,
		@Body() body: UpdateDataLakeDto,
	) {
		return { success: true, data: await this.smartCity.updateDataLake(req.authContext!, id, body) }
	}

	@Delete('data-lakes/:id')
	async deleteDataLake(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.deleteDataLake(req.authContext!, id) }
	}

	@Post('data-lakes/:id/disconnect')
	async disconnectDataLake(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.disconnectDataLake(req.authContext!, id) }
	}

	@Post('data-lakes/:id/test')
	async testDataLake(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.testDataLake(req.authContext!, id) }
	}

	@Get('models')
	async listModels(@Req() req: AuthenticatedRequest) {
		return { success: true, data: await this.smartCity.listModelArtifacts(req.authContext!) }
	}

	@Get('research-models')
	async listResearchModels() {
		return { success: true, data: await this.smartCity.listResearchModels() }
	}

	@Get('model-training-runs')
	async listTrainingRuns(@Req() req: AuthenticatedRequest) {
		return { success: true, data: await this.smartCity.listTrainingRuns(req.authContext!) }
	}

	@Post('models/train')
	async trainModel(@Req() req: AuthenticatedRequest, @Body() body: TrainModelDto) {
		return { success: true, data: await this.smartCity.trainModel(req.authContext!, body) }
	}

	@Post('models/:id/promote')
	async promoteModel(@Req() req: AuthenticatedRequest, @Param('id') id: string) {
		return { success: true, data: await this.smartCity.promoteModel(req.authContext!, id) }
	}

	@Post('pipelines/:pipelineId/models/:modelId/deploy')
	async deployModel(
		@Req() req: AuthenticatedRequest,
		@Param('pipelineId') pipelineId: string,
		@Param('modelId') modelId: string,
	) {
		return {
			success: true,
			data: await this.smartCity.deployModel(req.authContext!, pipelineId, modelId),
		}
	}

	@Post('pipelines/:pipelineId/research-models/:run/deploy')
	async deployResearchModel(
		@Req() req: AuthenticatedRequest,
		@Param('pipelineId') pipelineId: string,
		@Param('run') run: string,
	) {
		return {
			success: true,
			data: await this.smartCity.deployResearchModel(req.authContext!, pipelineId, run),
		}
	}

	@Post('pipelines/:pipelineId/test-processing')
	async testProcessing(
		@Req() req: AuthenticatedRequest,
		@Param('pipelineId') pipelineId: string,
		@Body() body: TestProcessingDto,
	) {
		return {
			success: true,
			data: await this.smartCity.testProcessing(req.authContext!, pipelineId, body.payload),
		}
	}
}
