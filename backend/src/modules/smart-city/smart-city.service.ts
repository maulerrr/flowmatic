import {
	BadRequestException,
	ForbiddenException,
	Injectable,
	Logger,
	NotFoundException,
	OnModuleInit,
	OnModuleDestroy,
} from '@nestjs/common'
import { Interval } from '@nestjs/schedule'
import { Prisma, SensorSource, SmartCityPipeline } from '@prisma/client'
import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'crypto'
import { existsSync, readFileSync, readdirSync } from 'fs'
import { mkdir, readdir, writeFile } from 'fs/promises'
import { basename, resolve } from 'path'
import { AppConfigService } from 'src/common/config/config.service'
import { BossService } from 'src/common/queue/boss.service'
import { parseCsvBuffer } from 'src/common/utils/csv-parser.util'
import { PrismaService } from 'src/prisma/prisma.service'
import { ExportService } from '../export/export.service'
import { ExportAdapterType, ExportResult } from '../export/types/export.types'
import { S3ConnectionOptions, StorageService } from '../storage/storage.service'
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
import { UpdateDataLakeDto } from './dto/update-data-lake.dto'
import { UpdateExportTargetDto } from './dto/update-export-target.dto'
import { UpdateSensorSourceDto } from './dto/update-sensor-source.dto'
import { UpdateSmartCityPipelineDto } from './dto/update-smart-city-pipeline.dto'
import { StartPipelineDto, UpdatePipelineRuntimeDto } from './dto/update-pipeline-runtime.dto'
import { TrainModelDto } from './dto/train-model.dto'
import { HuggingFaceIntegrationService } from '../integrations/huggingface.integration.service'
import { PipelineAutoRoutingService } from './pipeline-auto-routing.service'
import { PipelineModelRouterService } from './pipeline-model-router.service'
import { CoreUnitStreamConfig, ModelRoutingDecision } from './pipeline-model-router.types'

type AuthScope = {
	userId: string
	organizationId: string
	role: 'admin' | 'member' | 'viewer'
}

type StreamListener = (event: unknown) => void
type ResearchModelInfo = {
	id: string
	run: string
	name: string
	kind: unknown
	dataset: unknown
	metrics: unknown
	production: unknown
	localPath: string
	hasTorchScript: boolean
	hasSafetensors: boolean
}
type ExternalSocketState = {
	socket: any
	intentionalClose: boolean
	reconnectTimer?: NodeJS.Timeout
}
type PipelineRuntimeConfig = {
	lakeWriteMode: 'append' | 'object'
	sourcePollIntervalMs: number
	exportCadenceSeconds: number
	startedAt: string | null
	pausedAt: string | null
	lastRunningSourceIds: string[]
}

type FederatedConfig = {
	enabled: boolean
	protocol: 'HTTP' | 'WEBSOCKET'
	endpoint: string
	projectId: string | null
	nodeId: string | null
	topic: string | null
	headers: Record<string, string>
	registerPayload: Record<string, unknown>
	apiKeyEncrypted: string | null
	status: 'DISCONNECTED' | 'CONNECTED' | 'ERROR'
	lastConnectedAt: string | null
	lastTestedAt: string | null
	lastError: string | null
	lastTestResult: string | null
	registrationId: string | null
	registeredAt: string | null
	lastDeliveryAt: string | null
	globalModelVersion: string | null
	currentRoundId: string | null
	rounds: FederatedRound[]
}

type FederatedRoundStatus = 'ACTIVE' | 'AWAITING_UPDATES' | 'AGGREGATING' | 'COMPLETED' | 'FAILED'

type FederatedParticipantUpdate = {
	nodeId: string
	submittedAt: string
	sampleCount: number | null
	checkpointUri: string | null
	metrics: Record<string, unknown>
	notes: string | null
}

type FederatedRound = {
	id: string
	name: string
	status: FederatedRoundStatus
	startedAt: string
	completedAt: string | null
	sampleCount: number | null
	metrics: Record<string, unknown>
	participants: FederatedParticipantUpdate[]
	aggregatedModelVersion: string | null
	aggregatedCheckpointUri: string | null
	aggregationMetrics: Record<string, unknown>
	summary: string | null
}

const json = (value: unknown | undefined): Prisma.InputJsonValue | undefined =>
	value as Prisma.InputJsonValue | undefined
const SMART_CITY_EXPORT_QUEUE = 'smart-city-stage-export'

@Injectable()
export class SmartCityService implements OnModuleInit, OnModuleDestroy {
	private readonly logger = new Logger(SmartCityService.name)
	private reportedMissingExportTargetTable = false
	private readonly exportTargetLocks = new Set<string>()
	private readonly streamListeners = new Map<string, Set<StreamListener>>()
	private readonly externalSourceSockets = new Map<string, ExternalSocketState>()
	private readonly federatedSockets = new Map<string, ExternalSocketState>()
	private readonly lastProcessingErrorByPipeline = new Map<string, { message: string; at: number }>()

	constructor(
		private readonly prisma: PrismaService,
		private readonly config: AppConfigService,
		private readonly storageService: StorageService,
		private readonly exportService: ExportService,
		private readonly huggingFaceIntegration: HuggingFaceIntegrationService,
		private readonly boss: BossService,
		private readonly modelRouter: PipelineModelRouterService,
		private readonly autoRouting: PipelineAutoRoutingService,
	) {}

	async onModuleInit() {
		if (!this.boss.instance) return
		await this.boss.subscribe(SMART_CITY_EXPORT_QUEUE, async job => {
			const data =
				job && typeof job === 'object' && 'data' in job
					? ((job as { data?: unknown }).data as Record<string, unknown> | undefined)
					: undefined
			const targetId = typeof data?.targetId === 'string' ? data.targetId : null
			const force = data?.force === true
			if (!targetId) return
			await this.executeExportTargetById(targetId, force)
		})
	}

	onModuleDestroy() {
		for (const sourceId of [...this.externalSourceSockets.keys()]) {
			this.closeTrackedSocket(this.externalSourceSockets, sourceId)
		}
		for (const pipelineId of [...this.federatedSockets.keys()]) {
			this.closeTrackedSocket(this.federatedSockets, pipelineId)
		}
	}

	async listPipelines(scope: AuthScope) {
		const pipelines = await this.prisma.smartCityPipeline.findMany({
			where: { organizationId: scope.organizationId, status: { not: 'ARCHIVED' } },
			include: {
				sensorSources: {
					orderBy: { createdAt: 'asc' },
				},
			},
			orderBy: { updatedAt: 'desc' },
		})
		return pipelines.map(pipeline => this.presentPipeline(pipeline))
	}

	async createPipeline(scope: AuthScope, input: CreateSmartCityPipelineDto) {
		this.requireWriter(scope)
		const name = input.name.trim()
		const graphJson = this.createDefaultGraph(name)

		const pipeline = await this.prisma.smartCityPipeline.create({
			data: {
				organizationId: scope.organizationId,
				createdByUserId: scope.userId,
				name,
				description: input.description?.trim(),
				graphJson,
				streamConfig: json(this.defaultStreamConfig()),
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(pipeline)
	}

	async getPipeline(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		return this.presentPipeline(pipeline)
	}

	async updatePipeline(scope: AuthScope, pipelineId: string, input: UpdateSmartCityPipelineDto) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)

		if (input.dataLakeConnectionId) {
			await this.requireDataLake(scope.organizationId, input.dataLakeConnectionId)
		}

		const current = await this.requirePipeline(scope.organizationId, pipelineId)
		const pipeline = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				name: input.name?.trim(),
				description: input.description?.trim(),
				status: input.status,
				streamConfig: input.streamConfig
					? json(this.mergeStreamConfig(current.streamConfig, input.streamConfig))
					: undefined,
				activeModelId: input.activeModelId,
				dataLakeConnectionId: input.dataLakeConnectionId,
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(pipeline)
	}

	async deletePipeline(scope: AuthScope, pipelineId: string) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { status: 'ARCHIVED' },
		})
		return { deleted: true }
	}

	async startPipeline(scope: AuthScope, pipelineId: string, input: StartPipelineDto = {}) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const runtime = this.getRuntimeConfig(pipeline.streamConfig)
		const sourcePollIntervalMs = input.sourcePollIntervalMs ?? runtime.sourcePollIntervalMs
		const resumeSourceIds = runtime.lastRunningSourceIds.length > 0 ? runtime.lastRunningSourceIds : []
		const sources = await this.prisma.sensorSource.findMany({
			where: { organizationId: scope.organizationId, pipelineId },
			orderBy: { createdAt: 'asc' },
		})
		const targetSources =
			resumeSourceIds.length > 0
				? sources.filter(source => resumeSourceIds.includes(source.id))
				: sources
		if (sourcePollIntervalMs !== runtime.sourcePollIntervalMs) {
			await this.prisma.sensorSource.updateMany({
				where: { organizationId: scope.organizationId, pipelineId },
				data: { pollIntervalMs: sourcePollIntervalMs },
			})
		}
		const updated = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				status: 'ACTIVE',
				streamConfig: json(
					this.mergeStreamConfig(pipeline.streamConfig, {
						runtime: {
							...runtime,
							sourcePollIntervalMs,
							startedAt: new Date().toISOString(),
							pausedAt: null,
							lastRunningSourceIds: targetSources.map(source => source.id),
						},
					}),
				),
			},
			include: { sensorSources: true },
		})
		for (const source of targetSources) {
			await this.startSource(scope, source.id)
		}
		return this.presentPipeline(updated)
	}

	async stopPipeline(scope: AuthScope, pipelineId: string) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const runningSources = await this.prisma.sensorSource.findMany({
			where: { organizationId: scope.organizationId, pipelineId, status: 'RUNNING' },
			select: { id: true },
		})
		for (const source of runningSources) {
			await this.stopSource(scope, source.id)
		}
		const runtime = this.getRuntimeConfig(pipeline.streamConfig)
		const updated = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				status: 'PAUSED',
				streamConfig: json(
					this.mergeStreamConfig(pipeline.streamConfig, {
						runtime: {
							...runtime,
							pausedAt: new Date().toISOString(),
							lastRunningSourceIds: runningSources.map(source => source.id),
						},
					}),
				),
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(updated)
	}

	async resumePipeline(scope: AuthScope, pipelineId: string) {
		return this.startPipeline(scope, pipelineId)
	}

	async updatePipelineRuntime(scope: AuthScope, pipelineId: string, input: UpdatePipelineRuntimeDto) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const runtime = this.getRuntimeConfig(pipeline.streamConfig)
		const nextRuntime: Partial<PipelineRuntimeConfig> = {}
		if (input.lakeWriteMode) nextRuntime.lakeWriteMode = input.lakeWriteMode
		if (input.sourcePollIntervalMs) nextRuntime.sourcePollIntervalMs = input.sourcePollIntervalMs
		if (input.exportCadenceSeconds) nextRuntime.exportCadenceSeconds = input.exportCadenceSeconds
		if (input.sourcePollIntervalMs && pipeline.status === 'ACTIVE') {
			await this.prisma.sensorSource.updateMany({
				where: { organizationId: scope.organizationId, pipelineId },
				data: { pollIntervalMs: input.sourcePollIntervalMs },
			})
		}
		if (input.exportCadenceSeconds) {
			await this.prisma.smartCityExportTarget.updateMany({
				where: { organizationId: scope.organizationId, pipelineId, isContinuous: true },
				data: { cadenceSeconds: input.exportCadenceSeconds },
			})
		}
		const updated = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				streamConfig: json(
					this.mergeStreamConfig(pipeline.streamConfig, {
						runtime: {
							...runtime,
							...nextRuntime,
						},
					}),
				),
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(updated)
	}

	async updateGraph(scope: AuthScope, pipelineId: string, graph: Record<string, unknown>) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		return this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { graphJson: json(graph) },
		})
	}

	async getN8nWorkflow(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const sources = await this.prisma.sensorSource.findMany({
			where: { pipelineId, organizationId: scope.organizationId },
			orderBy: { createdAt: 'asc' },
		})
		const dataLake = pipeline.dataLakeConnectionId
			? await this.prisma.dataLakeConnection.findFirst({
					where: { id: pipeline.dataLakeConnectionId, organizationId: scope.organizationId },
				})
			: null

		const nodes = [
			...sources.map((source, index) => ({
				id: source.id,
				name: source.name,
				type: source.type === 'WEBSOCKET' ? 'flowmatic.websocketSensor' : 'flowmatic.httpPollingSensor',
				position: [180, 120 + index * 120],
				parameters: {
					mode: source.mode,
					sensorKind: source.sensorKind,
					intervalMs: source.pollIntervalMs,
					endpoint: source.endpoint,
				},
			})),
			{
				id: `${pipeline.id}:processor`,
				name: 'Flowmatic Processing Unit',
				type: 'flowmatic.processingUnit',
				position: [560, 220],
				parameters: {
					activeModelId: pipeline.activeModelId,
					streamConfig: pipeline.streamConfig,
				},
			},
			{
				id: `${pipeline.id}:data-lake`,
				name: dataLake?.name ?? 'S3 Data Lake',
				type: 'flowmatic.s3DataLake',
				position: [920, 160],
				parameters: dataLake
					? {
							provider: dataLake.provider,
							bucket: dataLake.bucket,
							basePrefix: dataLake.basePrefix,
						}
					: {},
			},
			{
				id: `${pipeline.id}:models`,
				name: 'Federated Model Training',
				type: 'flowmatic.modelLearning',
				position: [920, 340],
				parameters: {
					activeModelId: pipeline.activeModelId,
					federated: this.getFederatedConfig(pipeline.streamConfig),
				},
			},
		]

		return {
			name: pipeline.name,
			active: pipeline.status === 'ACTIVE',
			nodes,
			connections: sources.reduce<Record<string, unknown>>((acc, source) => {
				acc[source.name] = {
					main: [[{ node: 'Flowmatic Processing Unit', type: 'main', index: 0 }]],
				}
				return acc
			}, {
				'Flowmatic Processing Unit': {
					main: [
						[
							{ node: dataLake?.name ?? 'S3 Data Lake', type: 'main', index: 0 },
							{ node: 'Models and Federated Learning', type: 'main', index: 0 },
						],
					],
				},
			}),
			settings: {
				organizationId: scope.organizationId,
				pipelineId: pipeline.id,
				generatedBy: 'flowmatic',
			},
		}
	}

	async exportPipelineStage(
		scope: AuthScope,
		pipelineId: string,
		input: ExportPipelineStageDto,
	): Promise<ExportResult & { stage: string; rowCount: number }> {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const limit = Math.min(Math.max(input.limit ?? 100, 1), 500)
		const rows = await this.loadPipelineStageRows(scope.organizationId, pipeline.id, input.stage, limit)
		const settings = input.settings ?? {}
		const result = await this.executeExportRun({
			organizationId: scope.organizationId,
			pipelineId: pipeline.id,
			stage: input.stage,
			adapterType: input.adapterType,
			rows,
			settings,
			saveCredentials: input.saveCredentials ?? false,
			fileName: this.resolveExportFileName(
				input.adapterType,
				settings,
				pipeline.name,
				input.stage,
			),
		})
		return {
			...result,
			stage: input.stage,
			rowCount: rows.length,
		}
	}

	async previewExportStage(
		scope: AuthScope,
		pipelineId: string,
		stage: 'raw' | 'cleaned' | 'business',
		limit = 10,
	) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		const capped = Math.min(Math.max(limit, 1), 50)
		const rows = await this.loadPipelineStageRows(scope.organizationId, pipelineId, stage, capped)
		return {
			stage,
			rowCount: rows.length,
			columns: rows.length > 0 ? Object.keys(rows[0]) : [],
			rows,
		}
	}

	async backfillPipeline(scope: AuthScope, pipelineId: string, input: BackfillPipelineDto) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		if (!pipeline.dataLakeConnectionId) {
			throw new BadRequestException('Connect a data lake before running backfill')
		}

		const limit = Math.min(Math.max(input.limit ?? 500, 1), 5000)
		const stage = input.stage ?? 'all'
		const dataLake = await this.requireConnectedDataLake(scope.organizationId, pipeline.dataLakeConnectionId)
		await this.ensureDataLakeReady(dataLake)

		const events = await this.prisma.sensorEvent.findMany({
			where: { organizationId: scope.organizationId, pipelineId },
			orderBy: { eventTime: 'desc' },
			take: limit,
			include: {
				source: {
					select: { id: true, name: true, type: true, sensorKind: true, mode: true },
				},
			},
		})
		const orderedEvents = [...events].reverse()
		const counts = { raw: 0, cleaned: 0, business: 0 }

		for (const event of orderedEvents) {
			const payload =
				event.payloadJson && typeof event.payloadJson === 'object' && !Array.isArray(event.payloadJson)
					? (event.payloadJson as Record<string, unknown>)
					: { value: event.payloadJson }
			const cleanedPayload = this.createCleanedPayload(payload)

			if (stage === 'raw' || stage === 'all') {
				await this.backfillRawEventToDataLake(dataLake, pipeline, event)
				counts.raw += 1
			}

			if (stage === 'cleaned' || stage === 'all') {
				await this.backfillCleanedRecordToDataLake(dataLake, pipeline, event, cleanedPayload)
				counts.cleaned += 1
			}

			if (stage === 'business' || stage === 'all') {
				const result = await this.runProcessingInference(
					pipeline.organizationId,
					pipeline.activeModelId,
					cleanedPayload,
				)
				await this.backfillBusinessRecordToDataLake(
					dataLake,
					pipeline,
					event,
					cleanedPayload,
					result,
				)
				counts.business += 1
			}
		}

		const completedAt = new Date().toISOString()
		this.publishPipelineEvent(pipeline.id, {
			type: 'backfill_completed',
			data: {
				pipelineId: pipeline.id,
				stage,
				limit,
				scannedEvents: orderedEvents.length,
				counts,
				completedAt,
			},
		})
		return {
			pipelineId: pipeline.id,
			stage,
			limit,
			scannedEvents: orderedEvents.length,
			counts,
			completedAt,
			message: `Backfilled ${orderedEvents.length} events into medallion storage`,
		}
	}

	async listExportTargets(scope: AuthScope, pipelineId: string) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		return this.prisma.smartCityExportTarget.findMany({
			where: { organizationId: scope.organizationId, pipelineId, status: { not: 'ARCHIVED' } },
			orderBy: [{ isContinuous: 'desc' }, { updatedAt: 'desc' }],
		})
	}

	async listExportRuns(scope: AuthScope, pipelineId: string) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		return this.prisma.smartCityExportRun.findMany({
			where: { organizationId: scope.organizationId, pipelineId },
			orderBy: { createdAt: 'desc' },
			take: 50,
		})
	}

	async createExportTarget(scope: AuthScope, pipelineId: string, input: CreateExportTargetDto) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		const settings = input.settings ?? {}
		if (input.saveCredentials) {
			await this.exportService.persistAdapterCredentials(
				scope.organizationId,
				input.adapterType as ExportAdapterType,
				settings,
			)
		}
		return this.prisma.smartCityExportTarget.create({
			data: {
				organizationId: scope.organizationId,
				pipelineId,
				createdByUserId: scope.userId,
				name: input.name.trim(),
				stage: input.stage,
				adapterType: input.adapterType,
				settingsJson: json(settings),
				saveCredentials: input.saveCredentials ?? false,
				isContinuous: input.isContinuous ?? false,
				cadenceSeconds: input.cadenceSeconds ?? 60,
			},
		})
	}

	async updateExportTarget(scope: AuthScope, targetId: string, input: UpdateExportTargetDto) {
		this.requireWriter(scope)
		const target = await this.requireExportTarget(scope.organizationId, targetId)
		const existingSettings =
			target.settingsJson && typeof target.settingsJson === 'object' && !Array.isArray(target.settingsJson)
				? (target.settingsJson as Record<string, unknown>)
				: {}
		const mergedSettings = input.settings
			? this.mergeExportTargetSettings(existingSettings, input.settings, target.adapterType as ExportAdapterType)
			: existingSettings
		const adapterType = (input.adapterType ?? target.adapterType) as ExportAdapterType
		if (input.saveCredentials ?? target.saveCredentials) {
			await this.exportService.persistAdapterCredentials(scope.organizationId, adapterType, mergedSettings)
		}
		return this.prisma.smartCityExportTarget.update({
			where: { id: target.id },
			data: {
				name: input.name?.trim(),
				stage: input.stage,
				adapterType: input.adapterType,
				settingsJson: input.settings ? json(mergedSettings) : undefined,
				saveCredentials: input.saveCredentials,
				isContinuous: input.isContinuous,
				cadenceSeconds: input.cadenceSeconds,
				status:
					input.isContinuous === false && target.status === 'ERROR'
						? 'PAUSED'
						: input.settings || input.adapterType
							? 'ACTIVE'
							: undefined,
				lastError: input.settings || input.adapterType ? null : undefined,
			},
		})
	}

	async runExportTarget(scope: AuthScope, targetId: string) {
		this.requireWriter(scope)
		const target = await this.requireExportTarget(scope.organizationId, targetId)
		await this.executeExportTargetById(target.id, true)
		return { completed: true, targetId: target.id }
	}

	async deleteExportTarget(scope: AuthScope, targetId: string) {
		this.requireWriter(scope)
		const target = await this.requireExportTarget(scope.organizationId, targetId)
		await this.prisma.smartCityExportTarget.update({
			where: { id: target.id },
			data: {
				status: 'ARCHIVED',
				isContinuous: false,
			},
		})
		return { deleted: true, id: target.id }
	}

	async connectFederated(scope: AuthScope, pipelineId: string, input: ConnectFederatedDto) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.createFederatedConfig(input, this.getFederatedConfig(pipeline.streamConfig))
		const result = await this.establishFederatedConnection(pipelineId, config, false)
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, {
			...config,
			enabled: true,
			status: 'CONNECTED',
			lastConnectedAt: new Date().toISOString(),
			lastTestedAt: new Date().toISOString(),
			lastTestResult: result.summary,
			lastError: null,
			registrationId: result.registrationId ?? config.registrationId,
			registeredAt: result.registrationId ? new Date().toISOString() : config.registeredAt,
			globalModelVersion: result.globalModelVersion ?? config.globalModelVersion,
		})
		this.publishPipelineEvent(pipelineId, {
			type: 'federated_status',
			data: { pipelineId, status: 'CONNECTED', endpoint: config.endpoint, protocol: config.protocol },
		})
		return {
			pipeline: updated,
			connection: {
				ok: true,
				protocol: config.protocol,
				endpoint: config.endpoint,
				summary: result.summary,
			},
		}
	}

	async testFederated(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		if (!config.endpoint) throw new BadRequestException('Configure a federated endpoint first')
		try {
			const result = await this.establishFederatedConnection(pipelineId, config, true)
			await this.persistFederatedConfig(scope.organizationId, pipelineId, {
				...config,
				status: 'CONNECTED',
				lastTestedAt: new Date().toISOString(),
				lastTestResult: result.summary,
				lastError: null,
				registrationId: result.registrationId ?? config.registrationId,
				registeredAt: result.registrationId ? new Date().toISOString() : config.registeredAt,
				globalModelVersion: result.globalModelVersion ?? config.globalModelVersion,
			})
			return {
				ok: true,
				protocol: config.protocol,
				endpoint: config.endpoint,
				summary: result.summary,
			}
		} catch (error) {
			await this.persistFederatedConfig(scope.organizationId, pipelineId, {
				...config,
				status: 'ERROR',
				lastTestedAt: new Date().toISOString(),
				lastError: error instanceof Error ? error.message : 'Federated connection test failed',
			})
			throw error
		}
	}

	async disconnectFederated(scope: AuthScope, pipelineId: string) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		this.closeTrackedSocket(this.federatedSockets, pipelineId)
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, {
			...config,
			enabled: false,
			status: 'DISCONNECTED',
			lastError: null,
		})
		this.publishPipelineEvent(pipelineId, {
			type: 'federated_status',
			data: { pipelineId, status: 'DISCONNECTED', endpoint: config.endpoint, protocol: config.protocol },
		})
		return updated
	}

	async listFederatedRounds(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		return {
			registrationId: config.registrationId,
			globalModelVersion: config.globalModelVersion,
			currentRoundId: config.currentRoundId,
			rounds: config.rounds,
		}
	}

	async startFederatedRound(scope: AuthScope, pipelineId: string, input: StartFederatedRoundDto) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		if (!config.enabled || !config.endpoint) {
			throw new BadRequestException('Connect a federated coordinator before starting a round')
		}
		if (config.currentRoundId) {
			throw new BadRequestException('Complete the active federated round before starting a new one')
		}

		const round: FederatedRound = {
			id: `round-${Date.now()}`,
			name: input.name?.trim() || `Round ${config.rounds.length + 1}`,
			status: 'ACTIVE',
			startedAt: new Date().toISOString(),
			completedAt: null,
			sampleCount: input.sampleCount ?? null,
			metrics: input.metrics ?? {},
			participants: [],
			aggregatedModelVersion: null,
			aggregatedCheckpointUri: null,
			aggregationMetrics: {},
			summary: null,
		}
		const nextConfig: FederatedConfig = {
			...config,
			currentRoundId: round.id,
			lastError: null,
			rounds: [round, ...config.rounds].slice(0, 20),
		}
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, nextConfig)
		await this.forwardFederatedPayload(updated, 'round_started', {
			roundId: round.id,
			name: round.name,
			sampleCount: round.sampleCount,
			metrics: round.metrics,
		})
		this.publishPipelineEvent(pipelineId, {
			type: 'federated_status',
			data: { pipelineId, status: 'ROUND_ACTIVE', roundId: round.id, name: round.name },
		})
		return { round, pipeline: updated }
	}

	async submitFederatedUpdate(
		scope: AuthScope,
		pipelineId: string,
		roundId: string,
		input: SubmitFederatedUpdateDto,
	) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		const round = config.rounds.find(item => item.id === roundId)
		if (!round) throw new NotFoundException('Federated round not found')
		if (round.status === 'COMPLETED') {
			throw new BadRequestException('This federated round is already completed')
		}
		const update: FederatedParticipantUpdate = {
			nodeId: config.nodeId ?? `node-${scope.organizationId}`,
			submittedAt: new Date().toISOString(),
			sampleCount: input.sampleCount ?? null,
			checkpointUri: input.checkpointUri?.trim() ?? null,
			metrics: input.metrics ?? {},
			notes: input.notes?.trim() ?? null,
		}
		const nextRounds = config.rounds.map(item => {
			if (item.id !== roundId) return item
			const participants = [...item.participants.filter(p => p.nodeId !== update.nodeId), update]
			return {
				...item,
				status: 'AWAITING_UPDATES' as FederatedRoundStatus,
				participants,
			}
		})
		const nextConfig: FederatedConfig = {
			...config,
			lastDeliveryAt: update.submittedAt,
			rounds: nextRounds,
		}
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, nextConfig)
		await this.forwardFederatedPayload(updated, 'round_update_submitted', {
			roundId,
			update,
		})
		return {
			round: nextRounds.find(item => item.id === roundId),
			pipeline: updated,
		}
	}

	async aggregateFederatedRound(
		scope: AuthScope,
		pipelineId: string,
		roundId: string,
		input: AggregateFederatedRoundDto,
	) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		const round = config.rounds.find(item => item.id === roundId)
		if (!round) throw new NotFoundException('Federated round not found')

		const aggregatedVersion =
			input.globalModelVersion?.trim() ||
			`global-${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`
		const nextRounds = config.rounds.map(item => {
			if (item.id !== roundId) return item
			return {
				...item,
				status: 'COMPLETED' as FederatedRoundStatus,
				completedAt: new Date().toISOString(),
				aggregatedModelVersion: aggregatedVersion,
				aggregatedCheckpointUri: input.checkpointUri?.trim() ?? null,
				aggregationMetrics: input.metrics ?? {},
				summary: input.summary?.trim() ?? null,
			}
		})
		const nextRound = nextRounds.find(item => item.id === roundId)!
		const nextConfig: FederatedConfig = {
			...config,
			globalModelVersion: aggregatedVersion,
			currentRoundId: config.currentRoundId === roundId ? null : config.currentRoundId,
			lastDeliveryAt: new Date().toISOString(),
			rounds: nextRounds,
		}
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, nextConfig)
		await this.forwardFederatedPayload(updated, 'round_aggregated', {
			roundId,
			globalModelVersion: aggregatedVersion,
			checkpointUri: input.checkpointUri?.trim() ?? null,
			metrics: input.metrics ?? {},
			summary: input.summary?.trim() ?? null,
		})
		this.publishPipelineEvent(pipelineId, {
			type: 'federated_status',
			data: {
				pipelineId,
				status: 'ROUND_COMPLETED',
				roundId,
				globalModelVersion: aggregatedVersion,
			},
		})
		return { round: nextRound, pipeline: updated }
	}

	async syncFederatedGlobalModel(
		scope: AuthScope,
		pipelineId: string,
		input: SyncFederatedGlobalModelDto,
	) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const config = this.getFederatedConfig(pipeline.streamConfig)
		if (!config.enabled || !config.endpoint) {
			throw new BadRequestException('Connect a federated coordinator first')
		}

		const response = await this.requestFederatedGlobalState(pipeline.id, config)
		const nextConfig: FederatedConfig = {
			...config,
			globalModelVersion: response.globalModelVersion ?? config.globalModelVersion,
			lastTestedAt: new Date().toISOString(),
			lastTestResult: response.summary,
			lastError: null,
			lastDeliveryAt: new Date().toISOString(),
			currentRoundId: response.currentRoundId ?? config.currentRoundId,
			rounds: input.includeRounds && response.rounds ? response.rounds : config.rounds,
		}
		const updated = await this.persistFederatedConfig(scope.organizationId, pipelineId, nextConfig)
		return {
			pipeline: updated,
			globalModelVersion: nextConfig.globalModelVersion,
			currentRoundId: nextConfig.currentRoundId,
			summary: response.summary,
			rounds: nextConfig.rounds,
		}
	}

	async listSources(scope: AuthScope, pipelineId: string) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		return this.prisma.sensorSource.findMany({
			where: { pipelineId, organizationId: scope.organizationId },
			orderBy: { createdAt: 'asc' },
		})
	}

	async createSource(scope: AuthScope, pipelineId: string, input: CreateSensorSourceDto) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		this.validateSourceInput(input)

		const mode = input.mode ?? 'SIMULATED'
		const sensorKind = input.sensorKind ?? 'iot'
		const type = input.type
		let endpoint = input.endpoint?.trim()
		let connectionConfig = { ...(input.connectionConfig ?? {}) }
		if (mode === 'SIMULATED') {
			endpoint = this.buildSimulatorEndpoint(type, sensorKind)
			if (type === 'HTTP_POLLING' && !connectionConfig.payloadPath) {
				connectionConfig = { ...connectionConfig, payloadPath: 'events' }
			}
		}

		const source = await this.prisma.sensorSource.create({
			data: {
				organizationId: scope.organizationId,
				pipelineId,
				name: input.name.trim(),
				type,
				sensorKind,
				mode,
				endpoint,
				pollIntervalMs: input.pollIntervalMs ?? (type === 'WEBSOCKET' ? 1000 : 5000),
				connectionConfig: json(connectionConfig),
				schemaJson: this.defaultSchema(sensorKind),
			},
		})

		await this.refreshPipelineGraph(pipelineId)
		return source
	}

	async updateSource(scope: AuthScope, sourceId: string, input: UpdateSensorSourceDto) {
		this.requireWriter(scope)
		const source = await this.requireSource(scope.organizationId, sourceId)
		this.validateSourceInput({
			type: input.type ?? source.type,
			mode: input.mode ?? source.mode,
			endpoint: input.endpoint ?? source.endpoint ?? undefined,
		})
		this.closeTrackedSocket(this.externalSourceSockets, source.id)
		const updated = await this.prisma.sensorSource.update({
			where: { id: sourceId },
			data: {
				name: input.name?.trim(),
				type: input.type,
				sensorKind: input.sensorKind,
				mode: input.mode,
				endpoint: input.endpoint?.trim(),
				pollIntervalMs: input.pollIntervalMs,
				connectionConfig: json(input.connectionConfig),
			},
		})
		await this.refreshPipelineGraph(source.pipelineId)
		return updated
	}

	async deleteSource(scope: AuthScope, sourceId: string) {
		this.requireWriter(scope)
		const source = await this.requireSource(scope.organizationId, sourceId)
		this.closeTrackedSocket(this.externalSourceSockets, source.id)
		await this.prisma.sensorSource.delete({ where: { id: sourceId } })
		await this.refreshPipelineGraph(source.pipelineId)
		return { deleted: true }
	}

	async startSource(scope: AuthScope, sourceId: string) {
		this.requireWriter(scope)
		const source = this.withResolvedSourceEndpoint(await this.requireSource(scope.organizationId, sourceId))
		await this.prisma.sensorSource.update({
			where: { id: source.id },
			data: { status: 'RUNNING', lastError: null },
		})
		if (this.usesRemoteIngest(source) && source.type === 'WEBSOCKET') {
			await this.ensureExternalSourceConnection({
				...source,
				status: 'RUNNING',
				lastError: null,
			})
		} else if (this.usesRemoteIngest(source) && source.type === 'HTTP_POLLING') {
			await this.pollExternalHttpSource({
				...source,
				status: 'RUNNING',
				lastError: null,
			})
		}
		return this.requireSource(scope.organizationId, sourceId)
	}

	async stopSource(scope: AuthScope, sourceId: string) {
		this.requireWriter(scope)
		await this.requireSource(scope.organizationId, sourceId)
		this.closeTrackedSocket(this.externalSourceSockets, sourceId)
		return this.prisma.sensorSource.update({
			where: { id: sourceId },
			data: { status: 'STOPPED' },
		})
	}

	async testSource(scope: AuthScope, sourceId: string) {
		const source = this.withResolvedSourceEndpoint(await this.requireSource(scope.organizationId, sourceId))
		if (source.type === 'HTTP_POLLING') {
			const events = await this.pollExternalHttpSource(source, true)
			return {
				ok: true,
				mode: source.mode,
				type: source.type,
				summary: `Fetched ${events.length} event${events.length === 1 ? '' : 's'} from ${source.endpoint}`,
				event: events[0] ?? null,
				eventCount: events.length,
			}
		}
		return this.testExternalWebSocketSource(source)
	}

	async listEvents(scope: AuthScope, pipelineId: string, limit = 50) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		return this.prisma.sensorEvent.findMany({
			where: { pipelineId, organizationId: scope.organizationId },
			orderBy: { eventTime: 'desc' },
			take: Math.min(Math.max(limit, 1), 200),
			include: {
				source: {
					select: { id: true, name: true, type: true, sensorKind: true, status: true },
				},
			},
		})
	}

	async pollHttpStream(scope: AuthScope, pipelineId: string) {
		await this.requirePipeline(scope.organizationId, pipelineId)
		const sources = await this.prisma.sensorSource.findMany({
			where: {
				organizationId: scope.organizationId,
				pipelineId,
				type: 'HTTP_POLLING',
				status: 'RUNNING',
			},
			orderBy: { createdAt: 'asc' },
		})
		const remoteSources = sources.filter(source => this.usesRemoteIngest(source))
		const generated = await Promise.all(
			remoteSources.map(source =>
				this.pollExternalHttpSource(this.withResolvedSourceEndpoint(source), true),
			),
		)
		return {
			generated: generated.flat(),
			latest: await this.listEvents(scope, pipelineId, 25),
			nextPollMs: remoteSources.length > 0 ? Math.min(...remoteSources.map(source => source.pollIntervalMs)) : null,
		}
	}

	async getSimulatorPresets() {
		const baseUrl = this.config.sensorSimulator.baseUrl.replace(/\/$/, '')
		try {
			const response = await fetch(`${baseUrl}/api/v1/presets`)
			if (response.ok) {
				return response.json()
			}
		} catch (error) {
			this.logger.warn(
				`Sensor simulator presets unavailable at ${baseUrl}: ${error instanceof Error ? error.message : 'unknown error'}`,
			)
		}
		return {
			baseUrl,
			defaultLocation: 'Astana',
			sensorKinds: ['iot', 'video', 'power', 'network', 'weather', 'parking', 'traffic'],
			presets: {
				iot: {
					httpPollUrl: `${baseUrl}/api/v1/poll?sensorKind=iot&limit=1`,
					websocketUrl: `${baseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=iot`,
				},
				traffic: {
					httpPollUrl: `${baseUrl}/api/v1/poll?sensorKind=traffic&limit=1`,
					websocketUrl: `${baseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=traffic&intervalMs=2000`,
				},
			},
			astanaTraffic: {
				description: 'Semi-synthetic Astana traffic WebSocket stream with latitude/longitude.',
				websocketUrl: `${baseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=traffic&intervalMs=2000`,
			},
		}
	}

	async streamSnapshot(scope: AuthScope, pipelineId: string) {
		const [pipeline, sources, events] = await Promise.all([
			this.requirePipeline(scope.organizationId, pipelineId),
			this.listSources(scope, pipelineId),
			this.listEvents(scope, pipelineId, 25),
		])
		return {
			type: 'snapshot',
			data: {
				pipeline,
				sources,
				events,
				serverTime: new Date().toISOString(),
			},
		}
	}

	async assertPipelineAccess(scope: AuthScope, pipelineId: string) {
		await this.requirePipeline(scope.organizationId, pipelineId)
	}

	subscribeToPipeline(pipelineId: string, listener: StreamListener) {
		const listeners = this.streamListeners.get(pipelineId) ?? new Set<StreamListener>()
		listeners.add(listener)
		this.streamListeners.set(pipelineId, listeners)
		return () => {
			listeners.delete(listener)
			if (listeners.size === 0) this.streamListeners.delete(pipelineId)
		}
	}

	async getDashboard(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const [sources, events, dataLakes] = await Promise.all([
			this.listSources(scope, pipelineId),
			this.listEvents(scope, pipelineId, 25),
			this.listDataLakes(scope),
		])
		const runningSources = sources.filter(source => source.status === 'RUNNING').length

		return {
			pipeline: this.presentPipeline(pipeline),
			sources,
			events,
			dataLakes,
			stats: {
				sourceCount: sources.length,
				runningSources,
				eventCount: await this.prisma.sensorEvent.count({
					where: { organizationId: scope.organizationId, pipelineId },
				}),
				lastEventAt: events[0]?.eventTime ?? null,
			},
		}
	}

	async getObservability(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const since = new Date(Date.now() - 24 * 60 * 60 * 1000)
		const [sources, recentEvents, exportRuns, trainingRuns] = await Promise.all([
			this.listSources(scope, pipelineId),
			this.prisma.sensorEvent.findMany({
				where: { organizationId: scope.organizationId, pipelineId, eventTime: { gte: since } },
				orderBy: { eventTime: 'desc' },
				take: 200,
			}),
			this.prisma.smartCityExportRun.findMany({
				where: { organizationId: scope.organizationId, pipelineId, createdAt: { gte: since } },
				orderBy: { createdAt: 'desc' },
				take: 50,
			}),
			this.prisma.modelTrainingRun.findMany({
				where: { organizationId: scope.organizationId, pipelineId, createdAt: { gte: since } },
				orderBy: { createdAt: 'desc' },
				take: 20,
			}),
		])
		const federated = this.getFederatedConfig(pipeline.streamConfig)
		const alerts: Array<{ severity: 'info' | 'warning' | 'error'; message: string }> = []
		if (sources.length === 0) alerts.push({ severity: 'warning', message: 'No sources connected' })
		if (sources.some(source => source.status === 'ERROR')) {
			alerts.push({ severity: 'error', message: 'At least one source is in error state' })
		}
		if (sources.length > 0 && sources.every(source => source.status !== 'RUNNING')) {
			alerts.push({ severity: 'warning', message: 'No sources are currently running' })
		}
		if (federated.status === 'ERROR') {
			alerts.push({ severity: 'error', message: federated.lastError || 'Federated coordinator error' })
		}
		if (exportRuns.some(run => run.status === 'FAILED')) {
			alerts.push({ severity: 'warning', message: 'Recent export runs contain failures' })
		}

		return {
			windowHours: 24,
			eventsLast24h: recentEvents.length,
			runningSources: sources.filter(source => source.status === 'RUNNING').length,
			sourceErrors: sources.filter(source => source.status === 'ERROR').length,
			exportSuccesses: exportRuns.filter(run => run.status === 'SUCCEEDED').length,
			exportFailures: exportRuns.filter(run => run.status === 'FAILED').length,
			trainingRuns: trainingRuns.length,
			federatedStatus: federated.status,
			activeRoundId: federated.currentRoundId,
			globalModelVersion: federated.globalModelVersion,
			alerts: alerts.slice(0, 6),
			lastEventAt: recentEvents[0]?.eventTime ?? null,
			lastExportAt: exportRuns[0]?.createdAt ?? null,
			lastTrainingAt: trainingRuns[0]?.createdAt ?? null,
		}
	}

	async listDataLakes(scope: AuthScope) {
		const records = await this.prisma.dataLakeConnection.findMany({
			where: { organizationId: scope.organizationId },
			orderBy: [{ isDefault: 'desc' }, { updatedAt: 'desc' }],
		})
		return records.map(record => this.presentDataLake(record))
	}

	async listDataLakeObjects(scope: AuthScope, pipelineId: string, stage?: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		if (!pipeline.dataLakeConnectionId) {
			return { stage: stage ?? 'all', objects: [] }
		}
		const dataLake = await this.requireDataLake(scope.organizationId, pipeline.dataLakeConnectionId)
		await this.ensureDataLakeReady(dataLake)
		const stages =
			stage && ['raw', 'cleaned', 'business'].includes(stage)
				? [stage as 'raw' | 'cleaned' | 'business']
				: (['raw', 'cleaned', 'business'] as const)
		const groups = await Promise.all(
			stages.map(async currentStage => {
				const prefix = this.buildDataLakeBrowsePrefix(
					dataLake,
					currentStage,
					{
						organizationId: scope.organizationId,
						pipelineId,
						modelId: pipeline.activeModelId ?? 'unassigned-model',
					},
					new Date(),
				)
				const objects = await this.storageService.listObjects(
					dataLake.bucket,
					this.createDataLakeConnectionOptions(dataLake),
					prefix,
					50,
				)
				return {
					stage: currentStage,
					prefix,
					objects,
				}
			}),
		)
		return {
			stage: stage ?? 'all',
			objects: groups,
		}
	}

	async createDataLake(scope: AuthScope, input: CreateDataLakeDto) {
		this.requireWriter(scope)
		if (input.isDefault) await this.clearDefaultDataLake(scope.organizationId)

		const record = await this.prisma.dataLakeConnection.create({
			data: {
				organizationId: scope.organizationId,
				name: input.name.trim(),
				provider: input.provider ?? 'CUSTOM_S3',
				bucket: input.bucket.trim(),
				region: input.region?.trim(),
				endpoint: input.endpoint?.trim(),
				basePrefix: input.basePrefix?.trim() ?? '',
				accessKeyEncrypted: input.accessKey ? this.encryptSecret(input.accessKey) : undefined,
				secretKeyEncrypted: input.secretKey ? this.encryptSecret(input.secretKey) : undefined,
				isDefault: input.isDefault ?? false,
				pathRulesJson: json(input.pathRules ?? this.defaultPathRules()),
			},
		})
		return this.presentDataLake(record)
	}

	async updateDataLake(scope: AuthScope, id: string, input: UpdateDataLakeDto) {
		this.requireWriter(scope)
		await this.requireDataLake(scope.organizationId, id)
		if (input.isDefault) await this.clearDefaultDataLake(scope.organizationId)

		const record = await this.prisma.dataLakeConnection.update({
			where: { id },
			data: {
				name: input.name?.trim(),
				provider: input.provider,
				bucket: input.bucket?.trim(),
				region: input.region?.trim(),
				endpoint: input.endpoint?.trim(),
				basePrefix: input.basePrefix?.trim(),
				accessKeyEncrypted: input.accessKey ? this.encryptSecret(input.accessKey) : undefined,
				secretKeyEncrypted: input.secretKey ? this.encryptSecret(input.secretKey) : undefined,
				isDefault: input.isDefault,
				pathRulesJson: json(input.pathRules),
				status: 'CONNECTED',
			},
		})
		return this.presentDataLake(record)
	}

	async deleteDataLake(scope: AuthScope, id: string) {
		this.requireWriter(scope)
		await this.requireDataLake(scope.organizationId, id)
		await this.prisma.smartCityPipeline.updateMany({
			where: { organizationId: scope.organizationId, dataLakeConnectionId: id },
			data: { dataLakeConnectionId: null },
		})
		await this.prisma.dataLakeConnection.delete({ where: { id } })
		return { deleted: true }
	}

	async disconnectDataLake(scope: AuthScope, id: string) {
		this.requireWriter(scope)
		await this.requireDataLake(scope.organizationId, id)
		const record = await this.prisma.dataLakeConnection.update({
			where: { id },
			data: { status: 'DISCONNECTED', isDefault: false },
		})
		return this.presentDataLake(record)
	}

	async testDataLake(scope: AuthScope, id: string) {
		this.requireWriter(scope)
		const dataLake = await this.requireDataLake(scope.organizationId, id)
		try {
			await this.ensureDataLakeReady(dataLake)
			await this.uploadDataLakeJson(dataLake, `${this.buildDataLakePrefix(dataLake, '_flowmatic/tests', {
				organizationId: scope.organizationId,
			})}${Date.now()}-connectivity.json`, {
				type: 'connectivity_test',
				organizationId: scope.organizationId,
				dataLakeId: dataLake.id,
				timestamp: new Date().toISOString(),
			})
			const record = await this.prisma.dataLakeConnection.update({
				where: { id },
				data: {
					lastTestedAt: new Date(),
					lastTestStatus: 'CONNECTED_AND_WRITABLE',
					status: 'CONNECTED',
				},
			})
			return this.presentDataLake(record)
		} catch (error) {
			const record = await this.prisma.dataLakeConnection.update({
				where: { id },
				data: {
					lastTestedAt: new Date(),
					lastTestStatus: error instanceof Error ? error.message.slice(0, 180) : 'TEST_FAILED',
					status: 'ERROR',
				},
			})
			return this.presentDataLake(record)
		}
	}

	async listModelArtifacts(scope: AuthScope) {
		return this.prisma.modelArtifact.findMany({
			where: { organizationId: scope.organizationId, status: { not: 'ARCHIVED' } },
			orderBy: { createdAt: 'desc' },
		})
	}

	async listTrainingRuns(scope: AuthScope) {
		return this.prisma.modelTrainingRun.findMany({
			where: { organizationId: scope.organizationId },
			include: { artifacts: true },
			orderBy: { createdAt: 'desc' },
			take: 50,
		})
	}

	async trainModel(scope: AuthScope, input: TrainModelDto) {
		this.requireWriter(scope)
		if (input.pipelineId) await this.requirePipeline(scope.organizationId, input.pipelineId)

		const datasetPath = this.resolveDatasetPath(input.datasetPath)
		const run = await this.prisma.modelTrainingRun.create({
			data: {
				organizationId: scope.organizationId,
				pipelineId: input.pipelineId,
				createdByUserId: scope.userId,
				name: input.name?.trim() || 'Astana smart city baseline',
				datasetPath,
				modelType: input.modelType ?? 'TRAFFIC_BASELINE',
				status: 'QUEUED',
			},
		})

		try {
			await this.prisma.modelTrainingRun.update({
				where: { id: run.id },
				data: { status: 'RUNNING', startedAt: new Date() },
			})

			const trained = await this.trainBaselineModel(scope.organizationId, run.id, datasetPath)
			const artifact = await this.prisma.modelArtifact.create({
				data: {
					organizationId: scope.organizationId,
					trainingRunId: run.id,
					name: `${run.name} v${trained.version}`,
					modelType: run.modelType,
					version: trained.version,
					status: 'LOCAL_ONLY',
					localPath: trained.artifactDir,
					featureSpec: json(trained.featureSpec),
					metricsJson: json(trained.metrics),
				},
			})

			const completedRun = await this.prisma.modelTrainingRun.update({
				where: { id: run.id },
				data: {
					status: 'SUCCEEDED',
					localArtifactPath: trained.artifactDir,
					datasetProfile: json(trained.datasetProfile),
					featureSpec: json(trained.featureSpec),
					metricsJson: json(trained.metrics),
					logsJson: json(trained.logs),
					finishedAt: new Date(),
				},
				include: { artifacts: true },
			})
			if (run.pipelineId) {
				const pipeline = await this.prisma.smartCityPipeline.findUnique({
					where: { id: run.pipelineId },
					select: { id: true, streamConfig: true },
				})
				if (pipeline) {
					await this.forwardFederatedPayload(pipeline, 'training_completed', {
						runId: completedRun.id,
						name: completedRun.name,
						modelType: completedRun.modelType,
						status: completedRun.status,
						metrics: completedRun.metricsJson as Record<string, unknown>,
					})
				}
			}
			return completedRun
		} catch (error) {
			return this.prisma.modelTrainingRun.update({
				where: { id: run.id },
				data: {
					status: 'FAILED',
					errorMessage: error instanceof Error ? error.message : 'Unknown model training error',
					finishedAt: new Date(),
				},
				include: { artifacts: true },
			})
		}
	}

	async promoteModel(scope: AuthScope, artifactId: string) {
		this.requireWriter(scope)
		const artifact = await this.requireModelArtifact(scope.organizationId, artifactId)
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: { organizationId: scope.organizationId, status: 'CONNECTED' },
			orderBy: [{ isDefault: 'desc' }, { updatedAt: 'desc' }],
		})
		if (!dataLake) {
			throw new BadRequestException('Connect a data lake before promoting model artifacts')
		}

		const cleanPrefix = dataLake.basePrefix.replace(/^\/+|\/+$/g, '')
		const key = [cleanPrefix, 'models', scope.organizationId, artifact.id, artifact.version]
			.filter(Boolean)
			.join('/')
		return this.prisma.modelArtifact.update({
			where: { id: artifact.id },
			data: {
				status: 'PROMOTED',
				s3Uri: `s3://${dataLake.bucket}/${key}/model.json`,
			},
		})
	}

	async deployModel(scope: AuthScope, pipelineId: string, artifactId: string) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		const artifact = await this.requireModelArtifact(scope.organizationId, artifactId)
		await this.prisma.modelArtifact.update({
			where: { id: artifact.id },
			data: { status: artifact.status === 'LOCAL_ONLY' ? 'DEPLOYED' : artifact.status },
		})
		const pipeline = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { activeModelId: artifact.id },
			include: { sensorSources: true },
		})
		return this.presentPipeline(pipeline)
	}

	async listResearchModels() {
		const checkpointsDir = this.resolveWorkspacePath('models', 'checkpoints')
		if (!existsSync(checkpointsDir)) return []
		const entries = await readdir(checkpointsDir, { withFileTypes: true })
		const models: ResearchModelInfo[] = []
		for (const entry of entries) {
			if (!entry.isDirectory()) continue
			const runDir = resolve(checkpointsDir, entry.name)
			const metadataPath = resolve(runDir, 'metadata.json')
			if (!existsSync(metadataPath)) continue
			const metadata = JSON.parse(readFileSync(metadataPath, 'utf8')) as Record<string, unknown>
			models.push({
				id: `research:${entry.name}`,
				run: entry.name,
				name: entry.name,
				kind: metadata.kind,
				dataset: metadata.dataset,
				metrics: metadata.metrics,
				production: metadata.production,
				localPath: runDir,
				hasTorchScript: existsSync(resolve(runDir, 'model.torchscript.pt')),
				hasSafetensors: existsSync(resolve(runDir, 'model.safetensors')),
			})
		}
		return models.sort((a, b) => a.name.localeCompare(b.name))
	}

	async deployResearchModel(scope: AuthScope, pipelineId: string, run: string) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		const models = await this.listResearchModels()
		const model = models.find(item => item.run === run)
		if (!model) throw new NotFoundException('Research model checkpoint not found')
		await this.huggingFaceIntegration.prefetchModel(scope.organizationId, { localRun: run })
		const pipeline = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { activeModelId: model.id },
			include: { sensorSources: true },
		})
		return this.presentPipeline(pipeline)
	}

	async deployHuggingFaceModel(scope: AuthScope, pipelineId: string, modelId: string) {
		this.requireWriter(scope)
		await this.requirePipeline(scope.organizationId, pipelineId)
		const normalized = modelId.trim()
		await this.huggingFaceIntegration.getModelDetails(scope.organizationId, normalized)
		await this.huggingFaceIntegration.prefetchModel(scope.organizationId, { modelId: normalized })
		const pipeline = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { activeModelId: `hf:${normalized}` },
			include: { sensorSources: true },
		})
		return this.presentPipeline(pipeline)
	}

	async testProcessing(scope: AuthScope, pipelineId: string, payload?: Record<string, unknown>) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const samplePayload =
			payload && Object.keys(payload).length > 0
				? payload
				: {
						averageSpeedKph: 48,
						vehicleCount: 92,
						Latitude: 51.12,
						Longitude: 71.45,
						Traffic_Density: 74,
					}
		const streamConfig = this.getCoreUnitStreamConfig(pipeline.streamConfig)
		const decision = this.modelRouter.resolveModel({
			mode: streamConfig.coreUnitMode ?? 'manual',
			manualModelId: pipeline.activeModelId,
			sensorKind: 'traffic',
			payload: samplePayload,
			policy: streamConfig.autoRoutingPolicy,
			anomalyDetection: streamConfig.anomalyDetection !== false,
		})
		if (!decision.modelId) {
			return { ...decision, skipped: true }
		}
		const result = await this.runProcessingInference(
			scope.organizationId,
			decision.modelId,
			samplePayload,
		)
		return { ...result, routing: decision }
	}

	async previewCoreUnitRouting(scope: AuthScope, pipelineId: string) {
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const streamConfig = this.getCoreUnitStreamConfig(pipeline.streamConfig)
		const sensorKinds = [...new Set((pipeline.sensorSources ?? []).map(source => source.sensorKind))]
		const rulePolicy = this.modelRouter.buildDefaultPolicy(sensorKinds)
		return {
			coreUnitMode: streamConfig.coreUnitMode ?? 'manual',
			sensorKinds,
			registryModelCount: this.modelRouter.getRegistryModelCount(),
			activeModelId: pipeline.activeModelId,
			autoRoutingPolicy: streamConfig.autoRoutingPolicy ?? null,
			rulePolicy,
			lastAutoResolution: streamConfig.lastAutoResolution ?? null,
		}
	}

	async buildCoreUnitAutoPolicy(scope: AuthScope, pipelineId: string) {
		this.requireWriter(scope)
		const pipeline = await this.requirePipeline(scope.organizationId, pipelineId)
		const sensorKinds = [...new Set((pipeline.sensorSources ?? []).map(source => source.sensorKind))]
		const policy = await this.autoRouting.buildPolicy(sensorKinds)
		for (const binding of policy.bindings) {
			if (!binding.modelId.startsWith('research:')) continue
			const localRun = binding.modelId.replace(/^research:/, '')
			try {
				await this.huggingFaceIntegration.prefetchModel(scope.organizationId, { localRun })
			} catch (error) {
				this.logger.warn(`Prefetch failed for ${binding.modelId}: ${error instanceof Error ? error.message : error}`)
			}
		}
		const nextStreamConfig = this.mergeStreamConfig(pipeline.streamConfig, {
			coreUnitMode: 'auto',
			autoRoutingPolicy: policy,
		})
		const updated = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				streamConfig: json(nextStreamConfig),
				activeModelId: policy.bindings[0]?.modelId ?? pipeline.activeModelId,
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(updated)
	}

	@Interval(1000)
	async generateRunningSourceEvents() {
		const sources = await this.prisma.sensorSource.findMany({
			where: { status: 'RUNNING', pipeline: { status: 'ACTIVE' } },
			take: 100,
		})
		const now = Date.now()
		const dueSources = sources.filter(source => {
			if (source.type === 'WEBSOCKET') return false
			if (!this.usesRemoteIngest(source)) return false
			const lastSeenAt = source.lastSeenAt?.getTime() ?? 0
			return now - lastSeenAt >= source.pollIntervalMs
		})
		await Promise.all(
			sources
				.filter(source => source.type === 'WEBSOCKET' && this.usesRemoteIngest(source))
				.map(source => this.ensureExternalSourceConnection(this.withResolvedSourceEndpoint(source))),
		)
		await Promise.all(
			dueSources.map(source => this.pollExternalHttpSource(this.withResolvedSourceEndpoint(source))),
		)
	}

	@Interval(5000)
	async scheduleContinuousExports() {
		const now = Date.now()
		let targets: Array<{ id: string; lastRunAt: Date | null; cadenceSeconds: number }> = []
		try {
			targets = await this.prisma.smartCityExportTarget.findMany({
				where: {
					isContinuous: true,
					status: { in: ['ACTIVE', 'ERROR'] },
				},
				select: {
					id: true,
					lastRunAt: true,
					cadenceSeconds: true,
				},
				take: 100,
			})
		} catch (error) {
			if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2021') {
				if (!this.reportedMissingExportTargetTable) {
					this.reportedMissingExportTargetTable = true
					this.logger.warn(
						'Smart city export target table missing. Run the Prisma migrations to enable continuous exports.',
					)
				}
				return
			}
			throw error
		}
		for (const target of targets) {
			const lastRunAt = target.lastRunAt?.getTime() ?? 0
			if (now - lastRunAt < target.cadenceSeconds * 1000) continue
			try {
				await this.executeExportTargetById(target.id, false)
			} catch (error) {
				this.logger.warn(
					`Continuous export failed for target ${target.id}: ${error instanceof Error ? error.message : error}`,
				)
			}
		}
	}

	private async emitSensorEvent(sourceId: string) {
		const source = await this.prisma.sensorSource.findUnique({ where: { id: sourceId } })
		if (!source) throw new NotFoundException('Sensor source not found')

		const payload = this.createPayload(source.sensorKind, source.type)
		return this.persistSensorEvent(source, payload)
	}

	private async persistSensorEvent(source: SensorSource, payload: Record<string, unknown>) {
		const event = await this.prisma.sensorEvent.create({
			data: {
				organizationId: source.organizationId,
				pipelineId: source.pipelineId,
				sourceId: source.id,
				eventTime: new Date(),
				sensorType: source.sensorKind,
				location: this.resolvePayloadLocation(source, payload),
				payloadJson: json(payload),
			},
			include: {
				source: {
					select: { id: true, name: true, type: true, sensorKind: true, status: true },
				},
			},
		})

		await this.prisma.sensorSource.update({
			where: { id: source.id },
			data: { lastSeenAt: event.eventTime, lastError: null, status: 'RUNNING' },
		})
		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: source.pipelineId },
			select: { status: true, streamConfig: true },
		})
		const pipelineActive = pipeline?.status === 'ACTIVE'
		if (pipelineActive) {
			void this.exportRawEventToDataLake(source, event, pipeline.streamConfig).catch(() => {})
			void this.runProcessingForEvent(source.pipelineId, event).catch(() => {})
		}
		await this.trimRecentEvents(source.organizationId, source.pipelineId)
		this.publishPipelineEvent(source.pipelineId, { type: 'sensor_event', data: event })
		return event
	}

	private async runProcessingForEvent(
		pipelineId: string,
		event: { payloadJson: unknown; sensorType?: string | null },
	) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({ where: { id: pipelineId } })
		if (!pipeline || pipeline.status !== 'ACTIVE') return
		const cleanedPayload = this.createCleanedPayload(event.payloadJson as Record<string, unknown>)
		const streamConfig = this.getCoreUnitStreamConfig(pipeline.streamConfig)
		const decision = this.modelRouter.resolveModel({
			mode: streamConfig.coreUnitMode ?? 'manual',
			manualModelId: pipeline.activeModelId,
			sensorKind: event.sensorType ?? 'generic',
			payload: cleanedPayload,
			policy: streamConfig.autoRoutingPolicy,
			anomalyDetection: streamConfig.anomalyDetection !== false,
		})

		await this.exportCleanedRecordToDataLake(
			{
				id: pipeline.id,
				organizationId: pipeline.organizationId,
				dataLakeConnectionId: pipeline.dataLakeConnectionId,
				activeModelId: decision.modelId ?? pipeline.activeModelId,
			},
			cleanedPayload,
			pipeline.streamConfig,
		)
		if (!decision.modelId) {
			this.publishProcessingOutcome(pipelineId, {
				skipped: true,
				reason: decision.reason,
				routing: decision,
			})
			return
		}

		const result = await this.runProcessingInference(
			pipeline.organizationId,
			decision.modelId,
			cleanedPayload,
		)
		const enriched = { ...result, routing: decision }
		this.publishProcessingOutcome(pipelineId, enriched)
		if (result.error) return

		if (streamConfig.coreUnitMode === 'auto') {
			void this.recordAutoResolution(pipelineId, pipeline.streamConfig, decision)
		}

		await this.exportProcessingResultToDataLake(
			{
				id: pipeline.id,
				organizationId: pipeline.organizationId,
				dataLakeConnectionId: pipeline.dataLakeConnectionId,
				activeModelId: decision.modelId,
			},
			cleanedPayload,
			enriched,
			pipeline.streamConfig,
		)
		await this.forwardFederatedPayload(pipeline, 'processing_result', enriched)
	}

	private getCoreUnitStreamConfig(value: unknown) {
		return this.normalizeStreamConfig(value) as CoreUnitStreamConfig & Record<string, unknown>
	}

	private async recordAutoResolution(
		pipelineId: string,
		currentStreamConfig: unknown,
		decision: ModelRoutingDecision,
	) {
		if (!decision.modelId) return
		const nextStreamConfig = this.mergeStreamConfig(currentStreamConfig, {
			lastAutoResolution: {
				modelId: decision.modelId,
				label: decision.label,
				reason: decision.reason,
				sensorKind: decision.profile.sensorKind,
				at: new Date().toISOString(),
			},
		})
		await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { streamConfig: json(nextStreamConfig) },
		})
	}

	private async runProcessingInference(
		organizationId: string,
		activeModelId: string | null,
		payload: Record<string, unknown>,
	) {
		if (!activeModelId) {
			return { activeModelId: null, skipped: true, reason: 'No model selected for processing unit' }
		}
		if (activeModelId.startsWith('hf:')) {
			const modelId = activeModelId.replace(/^hf:/, '')
			try {
				const output = await this.huggingFaceIntegration.runInference(organizationId, modelId, payload)
				return { activeModelId, provider: 'flowmatic-local', modelId, output }
			} catch (error) {
				return {
					activeModelId,
					provider: 'flowmatic-local',
					modelId,
					error: error instanceof Error ? error.message : 'Local model inference failed',
				}
			}
		}
		if (!activeModelId.startsWith('research:')) {
			return {
				activeModelId,
				skipped: true,
				reason: 'Select a Hugging Face model in the core unit configuration',
			}
		}
		const run = activeModelId.replace(/^research:/, '')
		try {
			const output = await this.huggingFaceIntegration.runLocalInference(organizationId, run, payload)
			return { activeModelId, provider: 'flowmatic-local', localRun: run, output }
		} catch (error) {
			return {
				activeModelId,
				provider: 'flowmatic-local',
				localRun: run,
				error: error instanceof Error ? error.message : 'Local model inference failed',
			}
		}
	}

	private publishPipelineEvent(pipelineId: string, payload: unknown) {
		const listeners = this.streamListeners.get(pipelineId)
		if (!listeners) return
		for (const listener of listeners) listener(payload)
	}

	private publishProcessingOutcome(pipelineId: string, result: Record<string, unknown>) {
		if (result.error) {
			const message = String(result.error)
			const previous = this.lastProcessingErrorByPipeline.get(pipelineId)
			if (previous?.message === message && Date.now() - previous.at < 60_000) return
			this.lastProcessingErrorByPipeline.set(pipelineId, { message, at: Date.now() })
			this.publishPipelineEvent(pipelineId, { type: 'processing_error', data: result })
			return
		}
		this.lastProcessingErrorByPipeline.delete(pipelineId)
		this.publishPipelineEvent(pipelineId, { type: 'processing_result', data: result })
	}

	private async trimRecentEvents(organizationId: string, pipelineId: string) {
		const keep = await this.prisma.sensorEvent.findMany({
			where: { organizationId, pipelineId },
			orderBy: { eventTime: 'desc' },
			skip: 500,
			select: { id: true },
		})
		if (keep.length === 0) return
		await this.prisma.sensorEvent.deleteMany({ where: { id: { in: keep.map(event => event.id) } } })
	}

	private async ensureExternalSourceConnection(source: SensorSource) {
		if (!this.usesRemoteIngest(source) || source.type !== 'WEBSOCKET') return
		const endpoint = this.resolveSourceEndpoint(source)
		if (!endpoint.trim()) {
			throw new BadRequestException(`Source "${source.name}" needs an endpoint for WebSocket ingest`)
		}
		if (this.externalSourceSockets.has(source.id)) return
		const socket = await this.openWebSocketConnection({
			tracker: this.externalSourceSockets,
			key: source.id,
			endpoint,
			headers: this.extractHeaders(source.connectionConfig),
			subscribeMessage: this.extractSocketMessage(source.connectionConfig, 'subscribeMessage'),
			onMessage: async raw => {
				const payloads = this.normalizeSourcePayloads(source, raw)
				for (const payload of payloads) {
					await this.persistSensorEvent(source, payload)
				}
			},
			onError: async message => {
				await this.markSourceError(source.id, message)
			},
			onClose: async intentionalClose => {
				if (intentionalClose) return
				await this.markSourceError(source.id, 'External WebSocket connection closed')
				const latest = await this.prisma.sensorSource.findUnique({ where: { id: source.id } })
				if (latest?.status === 'RUNNING') {
					this.scheduleReconnect(this.externalSourceSockets, source.id, () =>
						this.ensureExternalSourceConnection(latest),
					)
				}
			},
		})
		if (!socket) return
		await this.prisma.sensorSource.update({
			where: { id: source.id },
			data: { status: 'RUNNING', lastError: null },
		})
	}

	private async pollExternalHttpSource(source: SensorSource, force = false) {
		if (!this.usesRemoteIngest(source) || source.type !== 'HTTP_POLLING') return []
		const endpoint = this.resolveSourceEndpoint(source)
		if (!endpoint.trim()) {
			throw new BadRequestException(`Source "${source.name}" needs an endpoint for HTTP polling ingest`)
		}
		const response = await fetch(endpoint, {
			method: this.extractHttpMethod(source.connectionConfig),
			headers: this.extractHeaders(source.connectionConfig),
		})
		if (!response.ok) {
			const message = `HTTP ${response.status} while polling ${endpoint}`
			await this.markSourceError(source.id, message)
			throw new BadRequestException(message)
		}
		const raw = await response.text()
		const payloads = this.normalizeSourcePayloads(source, raw)
		const events = await Promise.all(payloads.map(payload => this.persistSensorEvent(source, payload)))
		if (events.length === 0 && !force) return []
		return events
	}

	private async testExternalWebSocketSource(source: SensorSource) {
		const endpoint = this.resolveSourceEndpoint(source)
		if (!endpoint.trim()) {
			throw new BadRequestException(`Source "${source.name}" needs an endpoint for WebSocket ingest`)
		}
		let messageCount = 0
		let sample: Record<string, unknown> | null = null
		await this.openWebSocketConnection({
			tracker: new Map<string, ExternalSocketState>(),
			key: source.id,
			endpoint,
			headers: this.extractHeaders(source.connectionConfig),
			subscribeMessage: this.extractSocketMessage(source.connectionConfig, 'subscribeMessage'),
			testOnly: true,
			onMessage: async raw => {
				messageCount += 1
				if (!sample) sample = this.normalizeSourcePayloads(source, raw)[0] ?? null
			},
		})
		return {
			ok: true,
			mode: source.mode,
			type: source.type,
			summary:
				messageCount > 0
					? `Connected and observed ${messageCount} message${messageCount === 1 ? '' : 's'}`
					: 'Connected successfully',
			event: sample,
			eventCount: messageCount,
		}
	}

	private async openWebSocketConnection(options: {
		tracker: Map<string, ExternalSocketState>
		key: string
		endpoint: string
		headers?: Record<string, string>
		subscribeMessage?: string | null
		onMessage?: (raw: string) => Promise<void> | void
		onError?: (message: string) => Promise<void> | void
		onClose?: (intentionalClose: boolean) => Promise<void> | void
		testOnly?: boolean
	}) {
		const WebSocketCtor = (globalThis as { WebSocket?: any }).WebSocket
		if (!WebSocketCtor) throw new BadRequestException('WebSocket runtime is not available in this environment')
		const state: ExternalSocketState = { socket: null, intentionalClose: false }
		const socket = new WebSocketCtor(options.endpoint, undefined, {
			headers: options.headers,
		})
		state.socket = socket
		if (!options.testOnly) options.tracker.set(options.key, state)

		await new Promise<void>((resolve, reject) => {
			let finished = false
			const timeout = setTimeout(() => {
				if (finished) return
				finished = true
				state.intentionalClose = true
				try {
					socket.close()
				} catch {}
				reject(new BadRequestException(`Timed out connecting to ${options.endpoint}`))
			}, options.testOnly ? 4000 : 10000)

			const done = (callback: () => void) => {
				if (finished) return
				finished = true
				clearTimeout(timeout)
				callback()
			}

			socket.onopen = () => {
				if (options.subscribeMessage) socket.send(options.subscribeMessage)
				if (options.testOnly) {
					setTimeout(() => {
						done(() => {
							state.intentionalClose = true
							try {
								socket.close()
							} catch {}
							resolve()
						})
					}, 1200)
					return
				}
				done(resolve)
			}
			socket.onmessage = (event: { data: unknown }) => {
				const raw = typeof event.data === 'string' ? event.data : JSON.stringify(event.data)
				void options.onMessage?.(raw)
			}
			socket.onerror = (event: { error?: Error }) => {
				const message = event?.error?.message ?? `Could not connect to ${options.endpoint}`
				void options.onError?.(message)
				done(() => reject(new BadRequestException(message)))
			}
			socket.onclose = () => {
				if (options.testOnly && !finished) {
					done(resolve)
					return
				}
				const tracked = options.tracker.get(options.key)
				if (tracked && tracked.socket === socket) options.tracker.delete(options.key)
				void options.onClose?.(state.intentionalClose)
			}
		}).catch(error => {
			if (!options.testOnly) options.tracker.delete(options.key)
			throw error
		})

		return socket
	}

	private async markSourceError(sourceId: string, message: string) {
		await this.prisma.sensorSource.update({
			where: { id: sourceId },
			data: { status: 'ERROR', lastError: message },
		})
	}

	private normalizeSourcePayloads(source: SensorSource, raw: string) {
		const parsed = this.parseExternalPayload(raw)
		const payloadPath = this.readString(source.connectionConfig, 'payloadPath')
		const selected = payloadPath ? this.extractPath(parsed, payloadPath) : parsed
		const rows = Array.isArray(selected) ? selected : [selected]
		return rows
			.filter(row => row !== null && row !== undefined)
			.map(row => this.normalizePayloadRecord(row, source))
	}

	private parseExternalPayload(raw: string) {
		try {
			return JSON.parse(raw) as unknown
		} catch {
			return { raw }
		}
	}

	private normalizePayloadRecord(value: unknown, source: SensorSource) {
		if (value && typeof value === 'object' && !Array.isArray(value)) {
			return { ...(value as Record<string, unknown>) }
		}
		return {
			location: this.readString(source.connectionConfig, 'defaultLocation') ?? 'External sensor',
			value,
		}
	}

	private resolvePayloadLocation(source: SensorSource, payload: Record<string, unknown>) {
		const configuredField = this.readString(source.connectionConfig, 'locationField')
		if (configuredField) {
			const value = this.extractPath(payload, configuredField)
			if (typeof value === 'string' && value.trim()) return value.trim()
		}
		if (typeof payload.location === 'string' && payload.location.trim()) return payload.location.trim()
		return 'Astana'
	}

	private async requirePipeline(organizationId: string, pipelineId: string) {
		const pipeline = await this.prisma.smartCityPipeline.findFirst({
			where: { id: pipelineId, organizationId, status: { not: 'ARCHIVED' } },
			include: { sensorSources: { orderBy: { createdAt: 'asc' } } },
		})
		if (!pipeline) throw new NotFoundException('Smart city pipeline not found')
		return this.presentPipeline(pipeline)
	}
	private async requireSource(organizationId: string, sourceId: string) {
		const source = await this.prisma.sensorSource.findFirst({
			where: { id: sourceId, organizationId },
		})
		if (!source) throw new NotFoundException('Sensor source not found')
		return source
	}

	private async requireExportTarget(organizationId: string, targetId: string) {
		const target = await this.prisma.smartCityExportTarget.findFirst({
			where: { id: targetId, organizationId, status: { not: 'ARCHIVED' } },
		})
		if (!target) throw new NotFoundException('Smart city export target not found')
		return target
	}

	private mergeExportTargetSettings(
		existing: Record<string, unknown>,
		incoming: Record<string, unknown>,
		adapterType: ExportAdapterType,
	) {
		const cleanedIncoming = Object.fromEntries(
			Object.entries(incoming).filter(
				([, value]) => value !== undefined && value !== null && String(value).trim().length > 0,
			),
		)
		const merged = { ...existing, ...cleanedIncoming }
		const sensitiveKeys =
			adapterType === ExportAdapterType.HUGGINGFACE
				? ['token']
				: adapterType === ExportAdapterType.POSTGRES
					? ['password']
					: adapterType === ExportAdapterType.MONGODB
						? ['uri']
						: []
		for (const key of sensitiveKeys) {
			if (!cleanedIncoming[key] && existing[key]) merged[key] = existing[key]
		}
		return merged
	}

	private async requireDataLake(organizationId: string, id: string) {
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: { id, organizationId },
		})
		if (!dataLake) throw new NotFoundException('Data lake connection not found')
		return dataLake
	}

	private async requireConnectedDataLake(organizationId: string, id: string) {
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: { id, organizationId, status: { not: 'DISCONNECTED' } },
		})
		if (!dataLake) throw new BadRequestException('Connected data lake not found')
		return dataLake
	}

	private async requireModelArtifact(organizationId: string, id: string) {
		const artifact = await this.prisma.modelArtifact.findFirst({
			where: { id, organizationId, status: { not: 'ARCHIVED' } },
		})
		if (!artifact) throw new NotFoundException('Model artifact not found')
		return artifact
	}

	private requireWriter(scope: AuthScope) {
		if (scope.role === 'viewer') {
			throw new ForbiddenException('Viewers cannot change smart city pipelines')
		}
	}

	private async clearDefaultDataLake(organizationId: string) {
		await this.prisma.dataLakeConnection.updateMany({
			where: { organizationId, isDefault: true },
			data: { isDefault: false },
		})
	}

	private async refreshPipelineGraph(pipelineId: string) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: pipelineId },
			include: { sensorSources: { orderBy: { createdAt: 'asc' } } },
		})
		if (!pipeline) return
		await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: { graphJson: this.createGraph(pipeline.name, pipeline.sensorSources) },
		})
	}

	private presentPipeline<T extends SmartCityPipeline & { sensorSources?: SensorSource[] }>(pipeline: T): T {
		const streamConfig = this.normalizeStreamConfig(pipeline.streamConfig)
		return {
			...pipeline,
			streamConfig: {
				...streamConfig,
				federated: {
					...streamConfig.federated,
					apiKeyEncrypted: undefined,
				},
			},
		}
	}

	private createDefaultGraph(name: string) {
		return this.createGraph(name, [])
	}

	private createGraph(name: string, sources: Array<{ id: string; name: string; type: string }>) {
		return {
			name,
			nodes: [
				...sources.map((source, index) => ({
					id: source.id,
					type: source.type,
					label: source.name,
					x: 120,
					y: 120 + index * 110,
				})),
				{ id: 'processing-unit', type: 'PROCESSING_UNIT', label: 'Flowmatic Engine', x: 460, y: 220 },
				{ id: 'data-lake', type: 'DATA_LAKE', label: 'S3 Data Lake', x: 820, y: 150 },
				{ id: 'models', type: 'MODELS', label: 'Federated Training', x: 820, y: 320 },
			],
			edges: sources.flatMap(source => [
				{ from: source.id, to: 'processing-unit' },
				{ from: 'processing-unit', to: 'data-lake' },
				{ from: 'processing-unit', to: 'models' },
			]),
		}
	}

	private defaultStreamConfig() {
		return {
			coreUnitMode: 'manual',
			anomalyDetection: true,
			schemaValidation: true,
			autoCleaning: true,
			throughputLimit: 5,
			encryptionLevel: 'Standard',
			runtime: this.defaultRuntimeConfig(),
			federated: this.defaultFederatedConfig(),
		}
	}

	private defaultRuntimeConfig(): PipelineRuntimeConfig {
		return {
			lakeWriteMode: 'append',
			sourcePollIntervalMs: 3000,
			exportCadenceSeconds: 60,
			startedAt: null,
			pausedAt: null,
			lastRunningSourceIds: [],
		}
	}

	private getRuntimeConfig(value: unknown): PipelineRuntimeConfig {
		const config = this.normalizeStreamConfig(value)
		const runtime =
			config.runtime && typeof config.runtime === 'object' && !Array.isArray(config.runtime)
				? (config.runtime as Record<string, unknown>)
				: {}
		return {
			...this.defaultRuntimeConfig(),
			lakeWriteMode: runtime.lakeWriteMode === 'object' ? 'object' : 'append',
			sourcePollIntervalMs:
				typeof runtime.sourcePollIntervalMs === 'number' && runtime.sourcePollIntervalMs >= 1000
					? runtime.sourcePollIntervalMs
					: 3000,
			exportCadenceSeconds:
				typeof runtime.exportCadenceSeconds === 'number' && runtime.exportCadenceSeconds >= 15
					? runtime.exportCadenceSeconds
					: 60,
			startedAt: typeof runtime.startedAt === 'string' ? runtime.startedAt : null,
			pausedAt: typeof runtime.pausedAt === 'string' ? runtime.pausedAt : null,
			lastRunningSourceIds: Array.isArray(runtime.lastRunningSourceIds)
				? runtime.lastRunningSourceIds.filter((item): item is string => typeof item === 'string')
				: [],
		}
	}

	private normalizeStreamConfig(value: unknown) {
		const config =
			value && typeof value === 'object' && !Array.isArray(value)
				? { ...(value as Record<string, unknown>) }
				: {}
		return {
			...this.defaultStreamConfig(),
			...config,
			federated: this.getFederatedConfig(config),
		}
	}

	private mergeStreamConfig(current: unknown, patch: Record<string, unknown>) {
		const currentConfig = this.normalizeStreamConfig(current)
		const nextConfig = { ...currentConfig, ...patch }
		const currentFederated = this.getFederatedConfig(currentConfig)
		const patchFederated =
			patch.federated && typeof patch.federated === 'object' && !Array.isArray(patch.federated)
				? (patch.federated as Record<string, unknown>)
				: {}
		const currentRuntime = this.getRuntimeConfig(currentConfig)
		const patchRuntime =
			patch.runtime && typeof patch.runtime === 'object' && !Array.isArray(patch.runtime)
				? (patch.runtime as Record<string, unknown>)
				: {}
		return {
			...nextConfig,
			runtime: {
				...currentRuntime,
				...patchRuntime,
			},
			federated: this.getFederatedConfig({ ...currentFederated, ...patchFederated }),
		}
	}

	private defaultFederatedConfig(): FederatedConfig {
		return {
			enabled: false,
			protocol: 'HTTP',
			endpoint: '',
			projectId: null,
			nodeId: null,
			topic: null,
			headers: {},
			registerPayload: {},
			apiKeyEncrypted: null,
			status: 'DISCONNECTED',
			lastConnectedAt: null,
			lastTestedAt: null,
			lastError: null,
			lastTestResult: null,
			registrationId: null,
			registeredAt: null,
			lastDeliveryAt: null,
			globalModelVersion: null,
			currentRoundId: null,
			rounds: [],
		}
	}

	private getFederatedConfig(value: unknown): FederatedConfig {
		const container =
			value && typeof value === 'object' && !Array.isArray(value)
				? (value as Record<string, unknown>)
				: {}
		const raw =
			container.federated && typeof container.federated === 'object' && !Array.isArray(container.federated)
				? (container.federated as Record<string, unknown>)
				: container
		return {
			...this.defaultFederatedConfig(),
			...raw,
			headers: this.sanitizeHeaders(raw.headers),
			registerPayload:
				raw.registerPayload && typeof raw.registerPayload === 'object' && !Array.isArray(raw.registerPayload)
					? { ...(raw.registerPayload as Record<string, unknown>) }
					: {},
			rounds: this.normalizeFederatedRounds(raw.rounds),
		}
	}

	private createFederatedConfig(input: ConnectFederatedDto, current: FederatedConfig): FederatedConfig {
		return {
			...current,
			enabled: true,
			protocol: input.protocol,
			endpoint: input.endpoint.trim(),
			projectId: input.projectId?.trim() ?? current.projectId,
			nodeId: input.nodeId?.trim() ?? current.nodeId,
			topic: input.topic?.trim() ?? current.topic,
			headers: this.sanitizeHeaders(input.headers ?? current.headers),
			registerPayload: input.registerPayload ?? current.registerPayload,
			apiKeyEncrypted: input.apiKey ? this.encryptSecret(input.apiKey) : current.apiKeyEncrypted,
		}
	}

	private async persistFederatedConfig(
		organizationId: string,
		pipelineId: string,
		config: FederatedConfig,
	) {
		const pipeline = await this.prisma.smartCityPipeline.findFirst({
			where: { id: pipelineId, organizationId, status: { not: 'ARCHIVED' } },
			include: { sensorSources: { orderBy: { createdAt: 'asc' } } },
		})
		if (!pipeline) throw new NotFoundException('Smart city pipeline not found')
		const updated = await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				streamConfig: json({
					...this.normalizeStreamConfig(pipeline.streamConfig),
					federated: config,
				}),
			},
			include: { sensorSources: true },
		})
		return this.presentPipeline(updated)
	}

	private sanitizeHeaders(value: unknown) {
		if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
		return Object.fromEntries(
			Object.entries(value as Record<string, unknown>)
				.filter(([, item]) => typeof item === 'string' && item.trim().length > 0)
				.map(([key, item]) => [key, String(item)]),
		)
	}

	private normalizeFederatedRounds(value: unknown): FederatedRound[] {
		if (!Array.isArray(value)) return []
		return value
			.filter(item => item && typeof item === 'object' && !Array.isArray(item))
			.map(item => {
				const raw = item as Record<string, unknown>
				const participants = Array.isArray(raw.participants)
					? raw.participants
							.filter(entry => entry && typeof entry === 'object' && !Array.isArray(entry))
							.map(entry => {
								const participant = entry as Record<string, unknown>
								return {
									nodeId: typeof participant.nodeId === 'string' ? participant.nodeId : 'unknown-node',
									submittedAt:
										typeof participant.submittedAt === 'string'
											? participant.submittedAt
											: new Date().toISOString(),
									sampleCount:
										typeof participant.sampleCount === 'number' ? participant.sampleCount : null,
									checkpointUri:
										typeof participant.checkpointUri === 'string' ? participant.checkpointUri : null,
									metrics:
										participant.metrics &&
										typeof participant.metrics === 'object' &&
										!Array.isArray(participant.metrics)
											? { ...(participant.metrics as Record<string, unknown>) }
											: {},
									notes: typeof participant.notes === 'string' ? participant.notes : null,
								} satisfies FederatedParticipantUpdate
							})
					: []
				return {
					id: typeof raw.id === 'string' ? raw.id : `round-${Date.now()}`,
					name: typeof raw.name === 'string' ? raw.name : 'Federated round',
					status: this.normalizeFederatedRoundStatus(raw.status),
					startedAt: typeof raw.startedAt === 'string' ? raw.startedAt : new Date().toISOString(),
					completedAt: typeof raw.completedAt === 'string' ? raw.completedAt : null,
					sampleCount: typeof raw.sampleCount === 'number' ? raw.sampleCount : null,
					metrics:
						raw.metrics && typeof raw.metrics === 'object' && !Array.isArray(raw.metrics)
							? { ...(raw.metrics as Record<string, unknown>) }
							: {},
					participants,
					aggregatedModelVersion:
						typeof raw.aggregatedModelVersion === 'string' ? raw.aggregatedModelVersion : null,
					aggregatedCheckpointUri:
						typeof raw.aggregatedCheckpointUri === 'string' ? raw.aggregatedCheckpointUri : null,
					aggregationMetrics:
						raw.aggregationMetrics &&
						typeof raw.aggregationMetrics === 'object' &&
						!Array.isArray(raw.aggregationMetrics)
							? { ...(raw.aggregationMetrics as Record<string, unknown>) }
							: {},
					summary: typeof raw.summary === 'string' ? raw.summary : null,
				} satisfies FederatedRound
			})
			.slice(0, 20)
	}

	private normalizeFederatedRoundStatus(value: unknown): FederatedRoundStatus {
		return ['ACTIVE', 'AWAITING_UPDATES', 'AGGREGATING', 'COMPLETED', 'FAILED'].includes(
			String(value),
		)
			? (String(value) as FederatedRoundStatus)
			: 'ACTIVE'
	}

	private async establishFederatedConnection(
		pipelineId: string,
		config: FederatedConfig,
		testOnly: boolean,
	) {
		if (!config.endpoint.trim()) throw new BadRequestException('Federated endpoint is required')
		if (config.protocol === 'HTTP') {
			const response = await fetch(config.endpoint, {
				method: 'POST',
				headers: this.createFederatedHeaders(config),
				body: JSON.stringify(this.createFederatedPayload(config, 'register', {})),
			})
			if (!response.ok) {
				throw new BadRequestException(`Federated endpoint returned HTTP ${response.status}`)
			}
			const body = await response.text()
			const parsed = this.safeParseJson(body)
			return {
				summary: body.trim() ? `HTTP ${response.status}: ${body.slice(0, 120)}` : `HTTP ${response.status}`,
				registrationId:
					parsed && typeof parsed.registrationId === 'string' ? parsed.registrationId : null,
				globalModelVersion:
					parsed && typeof parsed.globalModelVersion === 'string' ? parsed.globalModelVersion : null,
			}
		}
		await this.openWebSocketConnection({
			tracker: this.federatedSockets,
			key: pipelineId,
			endpoint: config.endpoint,
			headers: this.createFederatedHeaders(config),
			subscribeMessage: JSON.stringify(this.createFederatedPayload(config, 'register', {})),
			testOnly,
			onError: async message => {
				await this.markFederatedError(pipelineId, message)
			},
			onClose: async intentionalClose => {
				if (testOnly || intentionalClose) return
				await this.markFederatedError(pipelineId, 'Federated WebSocket connection closed')
				const pipeline = await this.prisma.smartCityPipeline.findUnique({ where: { id: pipelineId } })
				const latest = this.getFederatedConfig(pipeline?.streamConfig)
				if (latest.enabled && latest.endpoint) {
					this.scheduleReconnect(this.federatedSockets, pipelineId, () =>
						this.establishFederatedConnection(pipelineId, latest, false).then(() => undefined),
					)
				}
			},
		})
		return {
			summary: testOnly ? 'WebSocket handshake succeeded' : 'WebSocket connected',
			registrationId: null,
			globalModelVersion: null,
		}
	}

	private async forwardFederatedPayload(
		pipeline: Pick<SmartCityPipeline, 'id' | 'streamConfig'>,
		type: string,
		payload: Record<string, unknown>,
	) {
		const config = this.getFederatedConfig(pipeline.streamConfig)
		if (!config.enabled || !config.endpoint) return
		try {
			if (config.protocol === 'HTTP') {
				await fetch(config.endpoint, {
					method: 'POST',
					headers: this.createFederatedHeaders(config),
					body: JSON.stringify(this.createFederatedPayload(config, type, payload)),
				})
				return
			}
			if (!this.federatedSockets.has(pipeline.id)) {
				await this.establishFederatedConnection(pipeline.id, config, false)
			}
			const state = this.federatedSockets.get(pipeline.id)
			state?.socket?.send(JSON.stringify(this.createFederatedPayload(config, type, payload)))
		} catch (error) {
			await this.markFederatedError(
				pipeline.id,
				error instanceof Error ? error.message : 'Federated forwarding failed',
			)
		}
	}

	private async requestFederatedGlobalState(pipelineId: string, config: FederatedConfig): Promise<{
		summary: string
		globalModelVersion: string | null
		currentRoundId: string | null
		rounds: FederatedRound[] | null
	}> {
		if (config.protocol === 'HTTP') {
			const response = await fetch(config.endpoint, {
				method: 'POST',
				headers: this.createFederatedHeaders(config),
				body: JSON.stringify(this.createFederatedPayload(config, 'pull_global_model', {})),
			})
			if (!response.ok) {
				throw new BadRequestException(`Federated endpoint returned HTTP ${response.status}`)
			}
			const body = await response.text()
			const parsed = this.safeParseJson(body)
			return {
				summary: body.trim() ? `HTTP ${response.status}: ${body.slice(0, 120)}` : `HTTP ${response.status}`,
				globalModelVersion:
					parsed && typeof parsed.globalModelVersion === 'string' ? parsed.globalModelVersion : null,
				currentRoundId: parsed && typeof parsed.currentRoundId === 'string' ? parsed.currentRoundId : null,
				rounds: parsed ? this.normalizeFederatedRounds(parsed.rounds) : null,
			}
		}

		if (!this.federatedSockets.has(pipelineId)) {
			await this.establishFederatedConnection(pipelineId, config, false)
		}
		const state = this.federatedSockets.get(pipelineId)
		state?.socket?.send(JSON.stringify(this.createFederatedPayload(config, 'pull_global_model', {})))
		return {
			summary: 'WebSocket pull request sent',
			globalModelVersion: config.globalModelVersion,
			currentRoundId: config.currentRoundId,
			rounds: null,
		}
	}

	private createFederatedHeaders(config: FederatedConfig) {
		const headers: Record<string, string> = {
			'content-type': 'application/json',
			...config.headers,
		}
		const apiKey = config.apiKeyEncrypted ? this.decryptSecret(config.apiKeyEncrypted) : null
		if (apiKey) headers.authorization = `Bearer ${apiKey}`
		return headers
	}

	private safeParseJson(value: string) {
		if (!value.trim()) return null
		try {
			const parsed = JSON.parse(value)
			return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
				? (parsed as Record<string, unknown>)
				: null
		} catch {
			return null
		}
	}

	private createFederatedPayload(
		config: FederatedConfig,
		type: string,
		data: Record<string, unknown>,
	) {
		return {
			type,
			projectId: config.projectId,
			nodeId: config.nodeId,
			topic: config.topic,
			registrationId: config.registrationId,
			currentRoundId: config.currentRoundId,
			globalModelVersion: config.globalModelVersion,
			timestamp: new Date().toISOString(),
			data,
			registerPayload: config.registerPayload,
		}
	}

	private async markFederatedError(pipelineId: string, message: string) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({ where: { id: pipelineId } })
		if (!pipeline) return
		const config = this.getFederatedConfig(pipeline.streamConfig)
		await this.prisma.smartCityPipeline.update({
			where: { id: pipelineId },
			data: {
				streamConfig: json({
					...this.normalizeStreamConfig(pipeline.streamConfig),
					federated: {
						...config,
						status: 'ERROR',
						lastError: message,
						lastTestedAt: new Date().toISOString(),
					},
				}),
			},
		})
	}

	private async exportRawEventToDataLake(
		source: Pick<SensorSource, 'organizationId' | 'pipelineId' | 'id' | 'name'>,
		event: {
			id: string
			eventTime: Date
			sensorType: string
			payloadJson: unknown
			location: string | null
		},
		streamConfig: unknown,
	) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: source.pipelineId },
			select: { id: true, organizationId: true, dataLakeConnectionId: true, activeModelId: true },
		})
		if (!pipeline?.dataLakeConnectionId) return
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: {
				id: pipeline.dataLakeConnectionId,
				organizationId: source.organizationId,
				status: { not: 'DISCONNECTED' },
			},
		})
		if (!dataLake) return

		try {
			await this.ensureDataLakeReady(dataLake)
			const prefix = this.buildDataLakePrefix(dataLake, 'raw', {
				organizationId: source.organizationId,
				pipelineId: source.pipelineId,
			}, event.eventTime)
			await this.writeDataLakeRecord(
				dataLake,
				prefix,
				streamConfig,
				{
					type: 'raw_sensor_event',
					sourceId: source.id,
					sourceName: source.name,
					sensorType: event.sensorType,
					location: event.location,
					eventTime: event.eventTime.toISOString(),
					eventId: event.id,
					payload: event.payloadJson,
				},
				`${event.id}.json`,
			)
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Raw event export failed',
			)
		}
	}

	private async exportCleanedRecordToDataLake(
		pipeline: Pick<
			SmartCityPipeline,
			'id' | 'organizationId' | 'dataLakeConnectionId' | 'activeModelId'
		>,
		cleanedPayload: Record<string, unknown>,
		streamConfig: unknown,
	) {
		if (!pipeline.dataLakeConnectionId) return
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: {
				id: pipeline.dataLakeConnectionId,
				organizationId: pipeline.organizationId,
				status: { not: 'DISCONNECTED' },
			},
		})
		if (!dataLake) return

		try {
			await this.ensureDataLakeReady(dataLake)
			const now = new Date()
			const prefix = this.buildDataLakePrefix(
				dataLake,
				'cleaned',
				{
					organizationId: pipeline.organizationId,
					pipelineId: pipeline.id,
					modelId: pipeline.activeModelId ?? 'unassigned-model',
				},
				now,
			)
			const objectKey = `${now.toISOString().replace(/[:.]/g, '-')}-${createHash('sha1').update(JSON.stringify(cleanedPayload)).digest('hex').slice(0, 8)}.json`
			await this.writeDataLakeRecord(
				dataLake,
				prefix,
				streamConfig,
				{
					type: 'cleaned_event',
					pipelineId: pipeline.id,
					activeModelId: pipeline.activeModelId,
					timestamp: now.toISOString(),
					payload: cleanedPayload,
				},
				objectKey,
			)
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Cleaned event export failed',
			)
		}
	}

	private async exportProcessingResultToDataLake(
		pipeline: Pick<
			SmartCityPipeline,
			'id' | 'organizationId' | 'dataLakeConnectionId' | 'activeModelId'
		>,
		inputPayload: Record<string, unknown>,
		result: Record<string, unknown>,
		streamConfig: unknown,
	) {
		if (!pipeline.dataLakeConnectionId) return
		const dataLake = await this.prisma.dataLakeConnection.findFirst({
			where: {
				id: pipeline.dataLakeConnectionId,
				organizationId: pipeline.organizationId,
				status: { not: 'DISCONNECTED' },
			},
		})
		if (!dataLake) return

		try {
			await this.ensureDataLakeReady(dataLake)
			const now = new Date()
			const prefix = this.buildDataLakePrefix(
				dataLake,
				'business',
				{
					organizationId: pipeline.organizationId,
					pipelineId: pipeline.id,
					modelId: pipeline.activeModelId ?? 'unassigned-model',
				},
				now,
			)
			const objectKey = `${now.toISOString().replace(/[:.]/g, '-')}-${createHash('sha1').update(JSON.stringify(result)).digest('hex').slice(0, 8)}.json`
			await this.writeDataLakeRecord(
				dataLake,
				prefix,
				streamConfig,
				{
					type: 'business_result',
					pipelineId: pipeline.id,
					activeModelId: pipeline.activeModelId,
					timestamp: now.toISOString(),
					inputPayload,
					result,
				},
				objectKey,
			)
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Processing result export failed',
			)
		}
	}

	private async backfillRawEventToDataLake(
		dataLake: {
			id: string
			name: string
			bucket: string
			region: string | null
			endpoint: string | null
			accessKeyEncrypted: string | null
			secretKeyEncrypted: string | null
			basePrefix: string
			pathRulesJson: unknown
		},
		pipeline: Pick<SmartCityPipeline, 'id' | 'organizationId' | 'activeModelId'>,
		event: {
			id: string
			eventTime: Date
			sensorType: string
			location: string | null
			payloadJson: unknown
			source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
		},
	) {
		try {
			const prefix = this.buildDataLakePrefix(
				dataLake,
				'raw',
				{ organizationId: pipeline.organizationId, pipelineId: pipeline.id },
				event.eventTime,
			)
			const key = `${prefix}backfill-${event.eventTime.toISOString().replace(/[:.]/g, '-')}-${event.id}.json`
			await this.uploadDataLakeJson(dataLake, key, {
				type: 'raw_sensor_event_backfill',
				pipelineId: pipeline.id,
				sourceId: event.source?.id,
				sourceName: event.source?.name,
				sensorType: event.sensorType,
				location: event.location,
				eventTime: event.eventTime.toISOString(),
				payload: event.payloadJson,
			})
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Raw event backfill failed',
			)
			throw error
		}
	}

	private async backfillCleanedRecordToDataLake(
		dataLake: {
			id: string
			name: string
			bucket: string
			region: string | null
			endpoint: string | null
			accessKeyEncrypted: string | null
			secretKeyEncrypted: string | null
			basePrefix: string
			pathRulesJson: unknown
		},
		pipeline: Pick<SmartCityPipeline, 'id' | 'organizationId' | 'activeModelId'>,
		event: {
			id: string
			eventTime: Date
			sensorType: string
			location: string | null
			source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
		},
		cleanedPayload: Record<string, unknown>,
	) {
		try {
			const prefix = this.buildDataLakePrefix(
				dataLake,
				'cleaned',
				{
					organizationId: pipeline.organizationId,
					pipelineId: pipeline.id,
					modelId: pipeline.activeModelId ?? 'unassigned-model',
				},
				event.eventTime,
			)
			const key = `${prefix}backfill-${event.eventTime.toISOString().replace(/[:.]/g, '-')}-${event.id}.json`
			await this.uploadDataLakeJson(dataLake, key, {
				type: 'cleaned_event_backfill',
				pipelineId: pipeline.id,
				eventId: event.id,
				activeModelId: pipeline.activeModelId,
				sourceId: event.source?.id,
				sourceName: event.source?.name,
				sensorType: event.sensorType,
				location: event.location,
				eventTime: event.eventTime.toISOString(),
				payload: cleanedPayload,
			})
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Cleaned event backfill failed',
			)
			throw error
		}
	}

	private async backfillBusinessRecordToDataLake(
		dataLake: {
			id: string
			name: string
			bucket: string
			region: string | null
			endpoint: string | null
			accessKeyEncrypted: string | null
			secretKeyEncrypted: string | null
			basePrefix: string
			pathRulesJson: unknown
		},
		pipeline: Pick<SmartCityPipeline, 'id' | 'organizationId' | 'activeModelId'>,
		event: {
			id: string
			eventTime: Date
			sensorType: string
			location: string | null
			source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
		},
		inputPayload: Record<string, unknown>,
		result: Record<string, unknown>,
	) {
		try {
			const prefix = this.buildDataLakePrefix(
				dataLake,
				'business',
				{
					organizationId: pipeline.organizationId,
					pipelineId: pipeline.id,
					modelId: pipeline.activeModelId ?? 'unassigned-model',
				},
				event.eventTime,
			)
			const key = `${prefix}backfill-${event.eventTime.toISOString().replace(/[:.]/g, '-')}-${event.id}.json`
			await this.uploadDataLakeJson(dataLake, key, {
				type: 'business_result_backfill',
				pipelineId: pipeline.id,
				eventId: event.id,
				activeModelId: pipeline.activeModelId,
				sourceId: event.source?.id,
				sourceName: event.source?.name,
				sensorType: event.sensorType,
				location: event.location,
				eventTime: event.eventTime.toISOString(),
				inputPayload,
				result,
			})
		} catch (error) {
			await this.markDataLakeError(
				dataLake.id,
				error instanceof Error ? error.message : 'Business result backfill failed',
			)
			throw error
		}
	}

	private async loadPipelineStageRows(
		organizationId: string,
		pipelineId: string,
		stage: 'raw' | 'cleaned' | 'business',
		limit: number,
	) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: pipelineId },
			select: { id: true, organizationId: true, activeModelId: true },
		})
		if (!pipeline || pipeline.organizationId !== organizationId) {
			throw new NotFoundException('Smart city pipeline not found')
		}

		const events = await this.prisma.sensorEvent.findMany({
			where: { organizationId, pipelineId },
			orderBy: { eventTime: 'desc' },
			take: limit,
			include: {
				source: {
					select: { id: true, name: true, type: true, sensorKind: true, mode: true },
				},
			},
		})

		const ordered = [...events].reverse()
		if (stage === 'raw') {
			return ordered.map(event => this.createRawExportRow(event))
		}
		if (stage === 'cleaned') {
			return ordered.map(event => this.createCleanedExportRow(event))
		}

		const rows = await Promise.all(
			ordered.map(async event =>
				this.createBusinessExportRow(pipeline.organizationId, pipeline.activeModelId, event),
			),
		)
		return rows
	}

	private async loadPipelineStageRowsSince(
		organizationId: string,
		pipelineId: string,
		stage: 'raw' | 'cleaned' | 'business',
		limit: number,
		afterEventId?: string | null,
	) {
		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: pipelineId },
			select: { id: true, organizationId: true, activeModelId: true },
		})
		if (!pipeline || pipeline.organizationId !== organizationId) {
			throw new NotFoundException('Smart city pipeline not found')
		}
		const events = await this.prisma.sensorEvent.findMany({
			where: {
				organizationId,
				pipelineId,
				...(afterEventId ? { id: { gt: afterEventId } } : {}),
			},
			orderBy: { id: 'asc' },
			take: limit,
			include: {
				source: {
					select: { id: true, name: true, type: true, sensorKind: true, mode: true },
				},
			},
		})
		if (stage === 'raw') return events.map(event => this.createRawExportRow(event))
		if (stage === 'cleaned') return events.map(event => this.createCleanedExportRow(event))
		return Promise.all(
			events.map(async event =>
				this.createBusinessExportRow(pipeline.organizationId, pipeline.activeModelId, event),
			),
		)
	}

	private async enqueueExportTarget(targetId: string, force = false) {
		if (!this.boss.instance) {
			await this.executeExportTargetById(targetId, force)
			return
		}
		await this.boss.publish(SMART_CITY_EXPORT_QUEUE, { targetId, force })
	}

	private async executeExportTargetById(targetId: string, force = false) {
		if (this.exportTargetLocks.has(targetId)) return
		this.exportTargetLocks.add(targetId)
		try {
			await this.executeExportTargetByIdLocked(targetId, force)
		} finally {
			this.exportTargetLocks.delete(targetId)
		}
	}

	private async executeExportTargetByIdLocked(targetId: string, force = false) {
		const target = await this.prisma.smartCityExportTarget.findUnique({
			where: { id: targetId },
		})
		if (!target || target.status === 'ARCHIVED' || target.status === 'PAUSED') return

		const pipeline = await this.prisma.smartCityPipeline.findUnique({
			where: { id: target.pipelineId },
			select: { id: true, status: true, name: true },
		})
		if (!pipeline || (!force && pipeline.status !== 'ACTIVE')) return

		await this.prisma.smartCityExportTarget.update({
			where: { id: target.id },
			data: { status: 'ACTIVE', lastError: null },
		})

		const baseSettings =
			target.settingsJson && typeof target.settingsJson === 'object' && !Array.isArray(target.settingsJson)
				? (target.settingsJson as Record<string, unknown>)
				: {}
		const settings = target.isContinuous
			? {
					...baseSettings,
					ifExists: baseSettings.ifExists ?? 'append',
					...(target.adapterType === 'json' || target.adapterType === 'csv'
						? {
								appendKey:
									typeof baseSettings.appendKey === 'string' && baseSettings.appendKey.trim()
										? baseSettings.appendKey
										: `smart-city/${target.pipelineId}/${target.stage}/${target.id}.ndjson`,
							}
						: {}),
				}
			: baseSettings
		const fileName = this.resolveExportFileName(
			target.adapterType,
			settings,
			target.name,
			target.stage,
		)

		const batchLimit = 500
		const maxBatches = force ? 100 : 20
		let cursorEventId = target.lastCursorEventId
		if (!cursorEventId && target.lastCursorAt) {
			const anchor = await this.prisma.sensorEvent.findFirst({
				where: {
					organizationId: target.organizationId,
					pipelineId: target.pipelineId,
					eventTime: { lte: target.lastCursorAt },
				},
				orderBy: [{ eventTime: 'desc' }, { id: 'desc' }],
				select: { id: true },
			})
			cursorEventId = anchor?.id ?? null
		}
		let totalExported = 0
		let lastRunId: string | undefined
		let batches = 0

		for (let batchIndex = 0; batchIndex < maxBatches; batchIndex++) {
			const rows = await this.loadPipelineStageRowsSince(
				target.organizationId,
				target.pipelineId,
				target.stage as 'raw' | 'cleaned' | 'business',
				batchLimit,
				cursorEventId,
			)
			if (rows.length === 0) {
				if (batchIndex === 0) {
					await this.prisma.smartCityExportTarget.update({
						where: { id: target.id },
						data: { lastRunAt: new Date(), lastError: null, status: 'ACTIVE' },
					})
				}
				break
			}

			const result = await this.executeExportRun({
				organizationId: target.organizationId,
				pipelineId: target.pipelineId,
				stage: target.stage as 'raw' | 'cleaned' | 'business',
				adapterType: target.adapterType as ExportAdapterType,
				rows,
				settings,
				saveCredentials: target.saveCredentials,
				fileName,
				targetId: target.id,
			})

			totalExported += result.recordsExported
			lastRunId = result.metadata?.smartCityExportRunId as string | undefined
			cursorEventId = this.extractLastEventId(rows) ?? cursorEventId
			batches += 1

			await this.prisma.smartCityExportTarget.update({
				where: { id: target.id },
				data: {
					lastRunAt: new Date(),
					lastRunId,
					lastCursorEventId: cursorEventId,
					lastCursorAt: this.extractNewestEventTimestamp(rows) ?? target.lastCursorAt,
					lastError: null,
					status: 'ACTIVE',
				},
			})

			if (rows.length < batchLimit) break
		}

		if (totalExported > 0) {
			this.logger.log(
				`Export target ${target.name} (${target.adapterType}): ${totalExported} rows in ${batches} batch(es)`,
			)
		}
	}

	private async executeExportRun(input: {
		organizationId: string
		pipelineId: string
		stage: 'raw' | 'cleaned' | 'business'
		adapterType: ExportAdapterType
		rows: Record<string, unknown>[]
		settings: Record<string, unknown>
		saveCredentials: boolean
		fileName: string
		targetId?: string
	}) {
		const run = await this.prisma.smartCityExportRun.create({
			data: {
				organizationId: input.organizationId,
				pipelineId: input.pipelineId,
				targetId: input.targetId,
				stage: input.stage,
				adapterType: input.adapterType,
				status: 'RUNNING',
				rowCount: input.rows.length,
				startedAt: new Date(),
				metadata: json({
					fileName: input.fileName,
					previewRows: input.rows.slice(0, 5),
					previewColumns: input.rows.length > 0 ? Object.keys(input.rows[0]) : [],
				}),
			},
		})
		try {
			const result = await this.exportService.exportDataRows({
				organizationId: input.organizationId,
				adapterType: input.adapterType,
				fileName: input.fileName,
				rows: input.rows,
				settings: input.settings,
				saveCredentials: input.saveCredentials,
				referenceId: run.id,
			})
			await this.prisma.smartCityExportRun.update({
				where: { id: run.id },
				data: {
					status: 'SUCCEEDED',
					recordsExported: result.recordsExported,
					destination: result.destination,
					message: result.message,
					metadata: json({
						...(result.metadata ?? {}),
						smartCityExportRunId: run.id,
					}),
					finishedAt: new Date(),
				},
			})
			return {
				...result,
				metadata: {
					...(result.metadata ?? {}),
					smartCityExportRunId: run.id,
				},
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Smart city export failed'
			await this.prisma.smartCityExportRun.update({
				where: { id: run.id },
				data: {
					status: 'FAILED',
					errorMessage: message,
					finishedAt: new Date(),
				},
			})
			if (input.targetId) {
				await this.prisma.smartCityExportTarget.update({
					where: { id: input.targetId },
					data: { status: 'ERROR', lastError: message, lastRunAt: new Date() },
				})
			}
			throw error
		}
	}

	private resolveExportFileName(
		adapterType: string,
		settings: Record<string, unknown>,
		targetName: string,
		stage: string,
	): string {
		if (adapterType === ExportAdapterType.HUGGINGFACE || adapterType === 'huggingface') {
			const fromSettings = settings.fileName
			if (typeof fromSettings === 'string' && fromSettings.trim()) return fromSettings.trim()
			return `smart_city_${stage}.csv`
		}
		if (adapterType === ExportAdapterType.CSV || adapterType === 'csv') {
			return `${targetName.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${stage}.csv`
		}
		return `${targetName.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${stage}.json`
	}

	private extractLastEventId(rows: Record<string, unknown>[]) {
		for (let index = rows.length - 1; index >= 0; index -= 1) {
			const eventId = rows[index]?.eventId
			if (typeof eventId === 'string' && eventId.trim()) return eventId.trim()
		}
		return null
	}

	private extractNewestEventTimestamp(rows: Record<string, unknown>[]) {
		const timestamps = rows
			.map(row => {
				const value = row.eventTime
				if (typeof value !== 'string') return null
				const date = new Date(value)
				return Number.isNaN(date.getTime()) ? null : date
			})
			.filter((value): value is Date => Boolean(value))
		if (timestamps.length === 0) return null
		return timestamps.reduce((latest, current) => (current > latest ? current : latest))
	}

	private createRawExportRow(event: {
		id: string
		eventTime: Date
		sensorType: string
		location: string | null
		payloadJson: unknown
		source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
	}) {
		const payload =
			event.payloadJson && typeof event.payloadJson === 'object' && !Array.isArray(event.payloadJson)
				? (event.payloadJson as Record<string, unknown>)
				: { value: event.payloadJson }
		return this.prepareExportRow({
			stage: 'raw',
			eventId: event.id,
			eventTime: event.eventTime.toISOString(),
			sourceId: event.source?.id,
			sourceName: event.source?.name,
			sourceType: event.source?.type,
			sourceMode: event.source?.mode,
			sensorType: event.sensorType,
			location: event.location,
			...payload,
		})
	}

	private createCleanedExportRow(event: {
		id: string
		eventTime: Date
		sensorType: string
		location: string | null
		payloadJson: unknown
		source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
	}) {
		const payload =
			event.payloadJson && typeof event.payloadJson === 'object' && !Array.isArray(event.payloadJson)
				? (event.payloadJson as Record<string, unknown>)
				: { value: event.payloadJson }
		return this.prepareExportRow({
			stage: 'cleaned',
			eventId: event.id,
			eventTime: event.eventTime.toISOString(),
			sourceId: event.source?.id,
			sourceName: event.source?.name,
			sourceType: event.source?.type,
			sourceMode: event.source?.mode,
			sensorType: event.sensorType,
			location: event.location,
			...this.createCleanedPayload(payload),
		})
	}

	private async createBusinessExportRow(
		organizationId: string,
		activeModelId: string | null,
		event: {
			id: string
			eventTime: Date
			sensorType: string
			location: string | null
			payloadJson: unknown
			source: { id: string; name: string; type: string; sensorKind: string; mode: string } | null
		},
	) {
		const payload =
			event.payloadJson && typeof event.payloadJson === 'object' && !Array.isArray(event.payloadJson)
				? (event.payloadJson as Record<string, unknown>)
				: { value: event.payloadJson }
		const cleaned = this.createCleanedPayload(payload)
		const result = await this.runProcessingInference(organizationId, activeModelId, cleaned)
		return this.prepareExportRow({
			stage: 'business',
			eventId: event.id,
			eventTime: event.eventTime.toISOString(),
			sourceId: event.source?.id,
			sourceName: event.source?.name,
			sourceType: event.source?.type,
			sourceMode: event.source?.mode,
			sensorType: event.sensorType,
			location: event.location,
			activeModelId,
			...cleaned,
			businessResult: result,
		})
	}

	private createCleanedPayload(payload: Record<string, unknown>) {
		const cleanedEntries = Object.entries(payload)
			.filter(([, value]) => value !== null && value !== undefined && String(value).trim().length > 0)
			.map(([key, value]) => [key, this.normalizeCleanedValue(value)] as const)
		return Object.fromEntries(cleanedEntries)
	}

	private normalizeCleanedValue(value: unknown): unknown {
		if (typeof value === 'string') return value.trim()
		if (Array.isArray(value)) return value.map(item => this.normalizeCleanedValue(item))
		if (value && typeof value === 'object') {
			return Object.fromEntries(
				Object.entries(value as Record<string, unknown>).map(([key, item]) => [
					key,
					this.normalizeCleanedValue(item),
				]),
			)
		}
		return value
	}

	private prepareExportRow(row: Record<string, unknown>) {
		return Object.fromEntries(
			Object.entries(row).map(([key, value]) => [key, this.prepareExportValue(value)]),
		)
	}

	private prepareExportValue(value: unknown): unknown {
		if (value instanceof Date) return value.toISOString()
		if (Array.isArray(value)) return JSON.stringify(value)
		if (value && typeof value === 'object') return JSON.stringify(value)
		return value ?? null
	}

	private async ensureDataLakeReady(dataLake: {
		id: string
		name: string
		bucket: string
		region: string | null
		endpoint: string | null
		accessKeyEncrypted: string | null
		secretKeyEncrypted: string | null
	}) {
		await this.storageService.ensureBucketExists(
			dataLake.bucket,
			this.createDataLakeConnectionOptions(dataLake),
		)
	}

	private async uploadDataLakeJson(
		dataLake: {
			id: string
			name: string
			bucket: string
			region: string | null
			endpoint: string | null
			accessKeyEncrypted: string | null
			secretKeyEncrypted: string | null
		},
		key: string,
		payload: Record<string, unknown>,
	) {
		await this.storageService.uploadObject(
			{
				bucket: dataLake.bucket,
				key,
				body: JSON.stringify(payload, null, 2),
				contentType: 'application/json',
				metadata: {
					flowmatic: 'smart-city',
					dataLakeId: dataLake.id,
				},
			},
			this.createDataLakeConnectionOptions(dataLake),
		)

		await this.prisma.dataLakeConnection.update({
			where: { id: dataLake.id },
			data: {
				status: 'CONNECTED',
				lastTestStatus: 'CONNECTED_AND_WRITABLE',
			},
		})
	}

	private async writeDataLakeRecord(
		dataLake: {
			id: string
			name: string
			bucket: string
			region: string | null
			endpoint: string | null
			accessKeyEncrypted: string | null
			secretKeyEncrypted: string | null
		},
		prefix: string,
		streamConfig: unknown,
		payload: Record<string, unknown>,
		objectKey: string,
	) {
		const runtime = this.getRuntimeConfig(streamConfig)
		const connection = this.createDataLakeConnectionOptions(dataLake)
		if (runtime.lakeWriteMode === 'append') {
			const key = `${prefix}stream.ndjson`
			await this.storageService.appendNdjsonLine(
				{
					bucket: dataLake.bucket,
					key,
					body: JSON.stringify(payload),
					contentType: 'application/x-ndjson',
					metadata: {
						flowmatic: 'smart-city',
						dataLakeId: dataLake.id,
					},
				},
				connection,
			)
		} else {
			await this.uploadDataLakeJson(dataLake, `${prefix}${objectKey}`, payload)
			return
		}
		await this.prisma.dataLakeConnection.update({
			where: { id: dataLake.id },
			data: {
				status: 'CONNECTED',
				lastTestStatus: 'CONNECTED_AND_WRITABLE',
			},
		})
	}

	private createDataLakeConnectionOptions(dataLake: {
		region: string | null
		endpoint: string | null
		accessKeyEncrypted: string | null
		secretKeyEncrypted: string | null
	}): S3ConnectionOptions {
		const accessKeyId =
			dataLake.accessKeyEncrypted ? this.decryptSecret(dataLake.accessKeyEncrypted) : this.config.s3.accessKeyId
		const secretAccessKey =
			dataLake.secretKeyEncrypted
				? this.decryptSecret(dataLake.secretKeyEncrypted)
				: this.config.s3.secretAccessKey
		if (!accessKeyId || !secretAccessKey) {
			throw new BadRequestException('Data lake credentials are missing')
		}
		return {
			endpoint: dataLake.endpoint ?? this.config.s3.accessEndpoint,
			region: dataLake.region ?? this.config.s3.region,
			accessKeyId,
			secretAccessKey,
			usePathStyle: this.config.s3.usePathStyle,
		}
	}

	private sanitizeDataLakeSegment(value: string) {
		return value
			.trim()
			.replace(/[^a-zA-Z0-9._-]/g, '-')
			.replace(/-+/g, '-')
			.replace(/^-+|-+$/g, '')
	}

	private sanitizeDataLakeValue(value: string) {
		const normalized = value.replace(/[\\/]+/g, '-')
		const sanitized = this.sanitizeDataLakeSegment(normalized)
		return sanitized || 'unknown'
	}

	private sanitizeDataLakePath(value: string) {
		return value
			.replace(/[\\]+/g, '/')
			.split('/')
			.map(segment => this.sanitizeDataLakeSegment(segment))
			.filter(Boolean)
			.join('/')
	}

	private buildDataLakePrefix(
		dataLake: { basePrefix: string; pathRulesJson: unknown },
		rule: 'raw' | 'cleaned' | 'business' | '_flowmatic/tests',
		context: {
			organizationId?: string
			pipelineId?: string
			modelId?: string
			version?: string
		},
		at: Date = new Date(),
	) {
		const pathRules =
			dataLake.pathRulesJson && typeof dataLake.pathRulesJson === 'object' && !Array.isArray(dataLake.pathRulesJson)
				? (dataLake.pathRulesJson as Record<string, unknown>)
				: {}
		const template =
			rule === '_flowmatic/tests'
				? '_flowmatic/tests/'
				: typeof pathRules[rule] === 'string'
					? String(pathRules[rule])
					: this.defaultPathRules()[rule]
		const replacements: Record<string, string> = {
			organizationId: this.sanitizeDataLakeValue(context.organizationId ?? 'unknown-org'),
			pipelineId: this.sanitizeDataLakeValue(context.pipelineId ?? 'unknown-pipeline'),
			modelId: this.sanitizeDataLakeValue(context.modelId ?? 'unknown-model'),
			version: this.sanitizeDataLakeValue(context.version ?? 'latest'),
			yyyy: this.sanitizeDataLakeValue(String(at.getUTCFullYear())),
			MM: this.sanitizeDataLakeValue(String(at.getUTCMonth() + 1).padStart(2, '0')),
			dd: this.sanitizeDataLakeValue(String(at.getUTCDate()).padStart(2, '0')),
			HH: this.sanitizeDataLakeValue(String(at.getUTCHours()).padStart(2, '0')),
		}
		const normalizedTemplate = template.replace(/[\\]+/g, '/')
		const resolved = Object.entries(replacements).reduce(
			(acc, [key, value]) => acc.replaceAll(`{${key}}`, value),
			normalizedTemplate,
		)
		const basePrefix = this.sanitizeDataLakePath(dataLake.basePrefix.replace(/^\/+|\/+$/g, ''))
		const cleanResolved = this.sanitizeDataLakePath(resolved.replace(/^\/+/, ''))
		return [basePrefix, cleanResolved].filter(Boolean).join('/').replace(/\/{2,}/g, '/') + '/'
	}

	private buildDataLakeBrowsePrefix(
		dataLake: { basePrefix: string; pathRulesJson: unknown },
		rule: 'raw' | 'cleaned' | 'business',
		context: {
			organizationId?: string
			pipelineId?: string
			modelId?: string
		},
		at: Date = new Date(),
	) {
		const pathRules =
			dataLake.pathRulesJson && typeof dataLake.pathRulesJson === 'object' && !Array.isArray(dataLake.pathRulesJson)
				? (dataLake.pathRulesJson as Record<string, unknown>)
				: {}
		const browseTemplate =
			typeof pathRules.browse === 'string'
				? String(pathRules.browse)
				: '{organizationId}/{pipelineId}/{stage}/{yyyy}/{MM}/{dd}/'
		const replacements: Record<string, string> = {
			organizationId: this.sanitizeDataLakeValue(context.organizationId ?? 'unknown-org'),
			pipelineId: this.sanitizeDataLakeValue(context.pipelineId ?? 'unknown-pipeline'),
			modelId: this.sanitizeDataLakeValue(context.modelId ?? 'unknown-model'),
			stage: rule,
			yyyy: this.sanitizeDataLakeValue(String(at.getUTCFullYear())),
			MM: this.sanitizeDataLakeValue(String(at.getUTCMonth() + 1).padStart(2, '0')),
			dd: this.sanitizeDataLakeValue(String(at.getUTCDate()).padStart(2, '0')),
		}
		const resolved = Object.entries(replacements).reduce(
			(acc, [key, value]) => acc.replaceAll(`{${key}}`, value),
			browseTemplate.replace(/[\\]+/g, '/'),
		)
		const basePrefix = this.sanitizeDataLakePath(dataLake.basePrefix.replace(/^\/+|\/+$/g, ''))
		const cleanResolved = this.sanitizeDataLakePath(resolved.replace(/^\/+/, ''))
		return [basePrefix, cleanResolved].filter(Boolean).join('/').replace(/\/{2,}/g, '/') + '/'
	}

	private async markDataLakeError(dataLakeId: string, message: string) {
		await this.prisma.dataLakeConnection.update({
			where: { id: dataLakeId },
			data: {
				status: 'ERROR',
				lastTestStatus: message.slice(0, 180),
			},
		})
	}

	private validateSourceInput(input: { type?: string; mode?: string; endpoint?: string }) {
		if (input.mode === 'EXTERNAL' && !input.endpoint?.trim()) {
			throw new BadRequestException('External sources require an endpoint')
		}
	}

	private usesRemoteIngest(source: Pick<SensorSource, 'mode'>) {
		return source.mode === 'EXTERNAL' || source.mode === 'SIMULATED'
	}

	private buildSimulatorEndpoint(type: string, sensorKind: string) {
		const baseUrl = this.config.sensorSimulator.baseUrl.replace(/\/$/, '')
		const kind = encodeURIComponent(sensorKind || 'iot')
		if (type === 'WEBSOCKET') {
			return `${baseUrl.replace(/^http/i, 'ws')}/ws?sensorKind=${kind}`
		}
		return `${baseUrl}/api/v1/poll?sensorKind=${kind}&limit=1`
	}

	private resolveSourceEndpoint(source: SensorSource) {
		if (source.mode === 'SIMULATED') {
			return this.buildSimulatorEndpoint(source.type, source.sensorKind)
		}
		return source.endpoint?.trim() ?? ''
	}

	private withResolvedSourceEndpoint(source: SensorSource): SensorSource {
		if (source.mode !== 'SIMULATED') return source
		const endpoint = this.buildSimulatorEndpoint(source.type, source.sensorKind)
		const connectionConfig =
			source.type === 'HTTP_POLLING'
				? {
						...(typeof source.connectionConfig === 'object' && source.connectionConfig && !Array.isArray(source.connectionConfig)
							? (source.connectionConfig as Record<string, unknown>)
							: {}),
						payloadPath:
							this.readString(source.connectionConfig, 'payloadPath') ?? 'events',
					}
				: source.connectionConfig
		return {
			...source,
			endpoint,
			connectionConfig: json(connectionConfig) as SensorSource['connectionConfig'],
		}
	}

	private closeTrackedSocket(tracker: Map<string, ExternalSocketState>, key: string) {
		const state = tracker.get(key)
		if (!state) return
		state.intentionalClose = true
		if (state.reconnectTimer) clearTimeout(state.reconnectTimer)
		try {
			state.socket?.close()
		} catch {}
		tracker.delete(key)
	}

	private scheduleReconnect(
		tracker: Map<string, ExternalSocketState>,
		key: string,
		callback: () => Promise<void>,
	) {
		const existing = tracker.get(key)
		const reconnectTimer = setTimeout(() => {
			tracker.delete(key)
			void callback()
		}, 5000)
		if (existing) {
			existing.socket = null
			existing.reconnectTimer = reconnectTimer
			return
		}
		tracker.set(key, { socket: null, intentionalClose: false, reconnectTimer })
	}

	private extractHeaders(value: unknown) {
		if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
		const rawHeaders = (value as Record<string, unknown>).headers
		return this.sanitizeHeaders(rawHeaders)
	}

	private extractSocketMessage(value: unknown, key: string) {
		const message = this.readString(value, key)
		if (!message) return null
		try {
			return JSON.stringify(JSON.parse(message))
		} catch {
			return message
		}
	}

	private extractHttpMethod(value: unknown) {
		const method = this.readString(value, 'method')?.toUpperCase()
		return method === 'POST' ? 'POST' : 'GET'
	}

	private readString(value: unknown, key: string) {
		if (!value || typeof value !== 'object' || Array.isArray(value)) return null
		const item = (value as Record<string, unknown>)[key]
		return typeof item === 'string' && item.trim() ? item.trim() : null
	}

	private extractPath(value: unknown, path: string) {
		return path.split('.').reduce<unknown>((current, segment) => {
			if (!current || typeof current !== 'object' || Array.isArray(current)) return undefined
			return (current as Record<string, unknown>)[segment]
		}, value)
	}

	private defaultSchema(sensorKind: string) {
		return {
			timestamp: 'eventTime',
			sensorKind,
			metrics: ['value', 'confidence'],
		}
	}

	private defaultPathRules() {
		return {
			raw: '{organizationId}/{pipelineId}/raw/{yyyy}/{MM}/{dd}/{HH}/',
			cleaned: '{organizationId}/{pipelineId}/cleaned/{yyyy}/{MM}/{dd}/{HH}/',
			business: '{organizationId}/{pipelineId}/business/{yyyy}/{MM}/{dd}/{HH}/',
			modelArtifacts: '{organizationId}/models/{modelId}/{version}/',
		}
	}

	private createPayload(sensorKind: string, transport: string) {
		const base = {
			location: 'Astana',
			transport,
			confidence: Number((0.82 + Math.random() * 0.17).toFixed(3)),
		}
		switch (sensorKind) {
			case 'video':
				return { ...base, vehicleCount: Math.floor(20 + Math.random() * 180), averageSpeedKph: Math.floor(20 + Math.random() * 55) }
			case 'power':
				return { ...base, loadPercent: Math.floor(35 + Math.random() * 60), voltage: Number((218 + Math.random() * 18).toFixed(1)) }
			case 'network':
				return { ...base, latencyMs: Math.floor(8 + Math.random() * 60), packetLossPercent: Number((Math.random() * 1.8).toFixed(2)) }
			case 'weather':
				return { ...base, temperatureC: Number((-8 + Math.random() * 35).toFixed(1)), windKph: Number((2 + Math.random() * 30).toFixed(1)) }
			case 'parking':
				return { ...base, occupancyPercent: Math.floor(20 + Math.random() * 78), openSpaces: Math.floor(Math.random() * 120) }
			default:
				return { ...base, airQualityIndex: Math.floor(20 + Math.random() * 130), pm25: Number((3 + Math.random() * 45).toFixed(1)) }
		}
	}

	private resolveDatasetPath(datasetPath?: string) {
		if (!datasetPath) return resolve(process.cwd(), '..', 'data', 'astana_synthetic_data.csv')
		const resolved = resolve(process.cwd(), datasetPath)
		if (!resolved.startsWith(resolve(process.cwd(), '..')) && !resolved.startsWith(process.cwd())) {
			throw new BadRequestException('Dataset path must be inside the project workspace')
		}
		return resolved
	}

	private resolveWorkspacePath(...parts: string[]) {
		const cwd = process.cwd()
		const fromCwd = resolve(cwd, ...parts)
		if (existsSync(fromCwd)) return fromCwd
		return resolve(cwd, '..', ...parts)
	}

	private async trainBaselineModel(organizationId: string, runId: string, datasetPath: string) {
		const parsed = parseCsvBuffer(readFileSync(datasetPath))
		if (parsed.rows.length < 10) {
			throw new BadRequestException('Dataset needs at least 10 rows for baseline training')
		}

		const numericColumns = parsed.columns.filter(column =>
			parsed.rows.some(row => Number.isFinite(Number(row[column]))),
		)
		const numericProfile = Object.fromEntries(
			numericColumns.map(column => [column, this.profileNumberColumn(parsed.rows, column)]),
		)
		const datasetProfile = {
			fileName: basename(datasetPath),
			rowCount: parsed.rows.length,
			columns: parsed.columns,
			numericColumns,
			timestampColumn: parsed.columns.find(column => /time|date|timestamp/i.test(column)) ?? null,
			locationColumns: parsed.columns.filter(column => /lat|lon|lng|location/i.test(column)),
			numericProfile,
			eventDistribution: this.distribution(parsed.rows, 'Event_Type'),
			severityDistribution: this.distribution(parsed.rows, 'Severity'),
		}

		const speed = numericProfile['Speed_kmh'] ?? this.profileNumberColumn(parsed.rows, numericColumns[0])
		const density =
			numericProfile['Traffic_Density'] ??
			this.profileNumberColumn(parsed.rows, numericColumns[numericColumns.length - 1])
		const metrics = {
			task: 'traffic_forecasting_and_anomaly_baseline',
			modelQuality: 'baseline',
			rowsTrained: parsed.rows.length,
			speedMean: speed.mean,
			speedStdDev: speed.stdDev,
			densityMean: density.mean,
			densityStdDev: density.stdDev,
			anomalyRules: {
				highSpeedKmh: Math.round(speed.mean + speed.stdDev * 2),
				lowSpeedKmh: Math.max(0, Math.round(speed.mean - speed.stdDev * 2)),
				highDensity: Number((density.mean + density.stdDev * 1.5).toFixed(2)),
			},
		}
		const featureSpec = {
			timestamp: datasetProfile.timestampColumn,
			entity: 'Event_ID',
			location: datasetProfile.locationColumns,
			numericFeatures: numericColumns,
			categoricalFeatures: parsed.columns.filter(column => !numericColumns.includes(column)),
			targetCandidates: ['Traffic_Density', 'Speed_kmh', 'Severity', 'Event_Type'].filter(column =>
				parsed.columns.includes(column),
			),
		}
		const model = {
			type: 'TRAFFIC_BASELINE',
			version: new Date().toISOString(),
			featureSpec,
			metrics,
			rules: {
				speedZScoreColumn: 'Speed_kmh',
				densityZScoreColumn: 'Traffic_Density',
				anomalyThresholdZScore: 2,
			},
		}
		const logs = [
			`Loaded ${parsed.rows.length} rows from ${basename(datasetPath)}`,
			`Detected numeric columns: ${numericColumns.join(', ')}`,
			'Built baseline traffic anomaly rules',
		]
		const version = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)
		const artifactDir = resolve(
			process.cwd(),
			'model-artifacts',
			'training',
			organizationId,
			runId,
		)
		await mkdir(artifactDir, { recursive: true })
		await Promise.all([
			writeFile(resolve(artifactDir, 'dataset-profile.json'), JSON.stringify(datasetProfile, null, 2)),
			writeFile(resolve(artifactDir, 'feature-spec.json'), JSON.stringify(featureSpec, null, 2)),
			writeFile(resolve(artifactDir, 'metrics.json'), JSON.stringify(metrics, null, 2)),
			writeFile(resolve(artifactDir, 'model.json'), JSON.stringify(model, null, 2)),
			writeFile(resolve(artifactDir, 'logs.json'), JSON.stringify(logs, null, 2)),
		])
		return { artifactDir, datasetProfile, featureSpec, metrics, logs, version }
	}

	private profileNumberColumn(rows: Record<string, unknown>[], column?: string) {
		if (!column) return { count: 0, min: 0, max: 0, mean: 0, stdDev: 0 }
		const values = rows
			.map(row => Number(row[column]))
			.filter(value => Number.isFinite(value))
		if (values.length === 0) return { count: 0, min: 0, max: 0, mean: 0, stdDev: 0 }
		const mean = values.reduce((sum, value) => sum + value, 0) / values.length
		const variance =
			values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length
		return {
			count: values.length,
			min: Math.min(...values),
			max: Math.max(...values),
			mean: Number(mean.toFixed(4)),
			stdDev: Number(Math.sqrt(variance).toFixed(4)),
		}
	}

	private distribution(rows: Record<string, unknown>[], column: string) {
		return rows.reduce<Record<string, number>>((acc, row) => {
			const key = String(row[column] ?? 'Unknown')
			acc[key] = (acc[key] ?? 0) + 1
			return acc
		}, {})
	}

	private encryptSecret(secret: string) {
		if (!secret.trim()) throw new BadRequestException('Secret value cannot be empty')
		const iv = randomBytes(12)
		const cipher = createCipheriv('aes-256-gcm', this.getEncryptionKey(), iv)
		const encrypted = Buffer.concat([cipher.update(secret, 'utf8'), cipher.final()])
		return [iv.toString('base64'), cipher.getAuthTag().toString('base64'), encrypted.toString('base64')].join(':')
	}

	private decryptSecret(secret: string) {
		const [iv, authTag, encrypted] = secret.split(':')
		if (!iv || !authTag || !encrypted) return null
		const decipher = createDecipheriv(
			'aes-256-gcm',
			this.getEncryptionKey(),
			Buffer.from(iv, 'base64'),
		)
		decipher.setAuthTag(Buffer.from(authTag, 'base64'))
		const decrypted = Buffer.concat([
			decipher.update(Buffer.from(encrypted, 'base64')),
			decipher.final(),
		])
		return decrypted.toString('utf8')
	}

	private getEncryptionKey() {
		return createHash('sha256').update(this.config.security.exportCredentialsSecret).digest()
	}

	private presentDataLake(record: {
		id: string
		organizationId: string
		name: string
		provider: string
		bucket: string
		region: string | null
		endpoint: string | null
		basePrefix: string
		accessKeyEncrypted: string | null
		secretKeyEncrypted: string | null
		isDefault: boolean
		status: string
		lastTestedAt: Date | null
		lastTestStatus: string | null
		pathRulesJson: unknown
		createdAt: Date
		updatedAt: Date
	}) {
		return {
			id: record.id,
			organizationId: record.organizationId,
			name: record.name,
			provider: record.provider,
			bucket: record.bucket,
			region: record.region,
			endpoint: record.endpoint,
			basePrefix: record.basePrefix,
			hasAccessKey: Boolean(record.accessKeyEncrypted),
			hasSecretKey: Boolean(record.secretKeyEncrypted),
			isDefault: record.isDefault,
			status: record.status,
			lastTestedAt: record.lastTestedAt,
			lastTestStatus: record.lastTestStatus,
			pathRules: record.pathRulesJson,
			createdAt: record.createdAt,
			updatedAt: record.updatedAt,
		}
	}
}
