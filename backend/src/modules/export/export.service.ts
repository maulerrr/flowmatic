import { Injectable, Logger, BadRequestException, NotFoundException } from '@nestjs/common'
import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'crypto'
import { PrismaService } from 'src/prisma/prisma.service'
import { StorageService } from '../storage/storage.service'
import { ExportAdapterRegistry } from './adapters/registry'
import { ExportConfig, ExportResult, ExportAdapterType } from './types/export.types'
import { PipelineRun, StorageFile } from '@prisma/client'
import { DataRow } from 'src/common/types/data.types'
import {
	createPaginationMeta,
	PaginationParamsFilter,
	PaginatedResponse,
} from 'src/common/utils/pagination.util'
import { parseCsvBuffer } from 'src/common/utils/csv-parser.util'
import { AppConfigService } from 'src/common/config/config.service'

/**
 * Service to orchestrate data exports to various destinations
 */
@Injectable()
export class ExportService {
	private readonly logger = new Logger(ExportService.name)

	constructor(
		private prisma: PrismaService,
		private storageService: StorageService,
		private adapterRegistry: ExportAdapterRegistry,
		private config: AppConfigService,
	) {}

	/**
	 * Get list of available export adapters
	 */
	getAvailableAdapters() {
		return this.adapterRegistry.getAdapterMetadata()
	}

	/**
	 * Get paginated preview data for a pipeline run
	 */
	async getPreviewData(
		runId: string,
		organizationId: string,
		pagination: PaginationParamsFilter,
	): Promise<
		PaginatedResponse<Record<string, unknown>> & { meta: { columns: string[]; fileName: string } }
	> {
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: runId },
			include: { resultFile: true, sourceFile: true },
		})

		if (!run) {
			throw new Error('Pipeline run not found')
		}

		if (run.organizationId !== organizationId) {
			throw new Error('Unauthorized access to run')
		}

		// Load all rows (from S3 or local storage)
		// Note: For very large files, this should be optimized to stream/seek,
		// but for MVP/Preview loading into memory matches current loadRunData logic.
		const allRows = await this.loadRunData(run)

		const { page = 1, pageSize = 10 } = pagination
		const totalCount = allRows.length
		const startIndex = (page - 1) * pageSize
		const endIndex = startIndex + pageSize

		const slicedData = allRows.slice(startIndex, endIndex)
		const columns = allRows.length > 0 ? Object.keys(allRows[0]) : []

		return {
			data: slicedData,
			pagination: createPaginationMeta(page, pageSize, totalCount),
			meta: {
				columns,
				fileName: run.resultFile?.fileName || run.sourceFileName,
			},
		}
	}

	/**
	 * Validate export configuration before execution
	 */
	async validateExportConfig(config: ExportConfig): Promise<{ valid: boolean; errors?: string[] }> {
		if (!this.adapterRegistry.hasAdapter(config.adapterType)) {
			return {
				valid: false,
				errors: [`Unknown adapter type: ${config.adapterType}`],
			}
		}

		const adapter = this.adapterRegistry.getAdapter(config.adapterType)!
		return adapter.validate(config.settings)
	}

	/**
	 * Export pipeline run results to specified destination
	 */
	async exportPipelineRun(
		pipelineRunId: string,
		organizationId: string,
		adapterType: ExportAdapterType,
		settings: Record<string, unknown>,
		saveCredentials: boolean = false,
	): Promise<ExportResult> {
		// Validate adapter exists
		if (!this.adapterRegistry.hasAdapter(adapterType)) {
			throw new BadRequestException(`Unknown adapter type: ${adapterType}`)
		}

		// Fetch pipeline run and result data
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: pipelineRunId },
			include: { organization: true, resultFile: true, sourceFile: true },
		})

		if (!run) {
			throw new NotFoundException('Pipeline run not found')
		}

		if (run.organizationId !== organizationId) {
			throw new BadRequestException('Unauthorized to export this run')
		}

		// Load cleaned data from result file (S3)
		const data = await this.loadRunData(run)

		const mergedSettings = await this.mergeSavedCredentials(
			organizationId,
			adapterType,
			settings,
			saveCredentials,
		)

		const exportConfig: ExportConfig = {
			adapterType,
			organizationId,
			pipelineRunId,
			fileName: run.sourceFileName,
			settings: mergedSettings,
			saveCredentials,
		}

		// Validate config
		const validation = await this.validateExportConfig(exportConfig)
		if (!validation.valid) {
			throw new BadRequestException(
				`Invalid export configuration: ${validation.errors?.join(', ')}`,
			)
		}

		// Execute export
		const adapter = this.adapterRegistry.getAdapter(adapterType)!

		try {
			const result = await adapter.export(data, exportConfig)

			if (saveCredentials) {
				await this.saveAdapterCredentials(organizationId, adapterType, mergedSettings)
			}

			// Record export in database (TODO: uncomment after Prisma generation)
			await this.recordExport(pipelineRunId, adapterType, result)

			return result
		} catch (error) {
			this.logger.error(
				`Export failed for run ${pipelineRunId} using adapter ${adapterType}`,
				error instanceof Error ? error.message : 'Unknown error',
			)
			throw error
		}
	}

	async exportDataRows(params: {
		organizationId: string
		adapterType: ExportAdapterType
		fileName: string
		rows: Record<string, unknown>[]
		settings?: Record<string, unknown>
		saveCredentials?: boolean
		referenceId?: string
	}): Promise<ExportResult> {
		if (!this.adapterRegistry.hasAdapter(params.adapterType)) {
			throw new BadRequestException(`Unknown adapter type: ${params.adapterType}`)
		}

		const mergedSettings = await this.mergeSavedCredentials(
			params.organizationId,
			params.adapterType,
			params.settings ?? {},
			params.saveCredentials ?? false,
		)

		const exportConfig: ExportConfig = {
			adapterType: params.adapterType,
			organizationId: params.organizationId,
			pipelineRunId: params.referenceId ?? `smart-city:${Date.now()}`,
			fileName: params.fileName,
			settings: mergedSettings,
			saveCredentials: params.saveCredentials,
		}

		const validation = await this.validateExportConfig(exportConfig)
		if (!validation.valid) {
			throw new BadRequestException(
				`Invalid export configuration: ${validation.errors?.join(', ')}`,
			)
		}

		const adapter = this.adapterRegistry.getAdapter(params.adapterType)!
		const preparedRows = params.rows.map(row => this.prepareRowForAdapter(row))

		const result = await adapter.export(preparedRows, exportConfig)
		if (params.saveCredentials) {
			await this.saveAdapterCredentials(
				params.organizationId,
				params.adapterType,
				mergedSettings,
			)
		}
		return result
	}

	/**
	 * Load data from pipeline run result file
	 */
	private async loadRunData(
		run: PipelineRun & { resultFile: StorageFile | null },
	): Promise<DataRow[]> {
		if (!run.resultFile) {
			throw new Error('No result file available for this pipeline run')
		}

		const key = run.resultFile.s3Key
		if (!key) {
			throw new Error('Result file has no S3 key')
		}

		try {
			const buffer = await this.storageService.downloadFileFromS3(
				key,
				this.storageService.defaultBucket,
			)
			const mime = run.resultFile.mimeType?.toLowerCase() || ''
			const fileName = run.resultFile.fileName?.toLowerCase() || ''

			// Decide parser based on mime or extension
			if (mime.includes('json') || fileName.endsWith('.json')) {
				return JSON.parse(buffer.toString('utf-8')) as DataRow[]
			}

			// Default: parse CSV
			return this.parseCsv(buffer)
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Unknown error'
			this.logger.error(`Failed to load run data from S3 (key: ${key}): ${errorMessage}`)
			throw new Error(`Failed to load run data: ${errorMessage}`)
		}
	}

	private parseCsv(buffer: Buffer): DataRow[] {
		return parseCsvBuffer(buffer).rows
	}

	private prepareRowForAdapter(row: Record<string, unknown>) {
		return Object.fromEntries(
			Object.entries(row).map(([key, value]) => [key, this.prepareValueForAdapter(value)]),
		)
	}

	private prepareValueForAdapter(value: unknown): unknown {
		if (value instanceof Date) return value.toISOString()
		if (Array.isArray(value)) return JSON.stringify(value)
		if (value && typeof value === 'object') return JSON.stringify(value)
		return value ?? null
	}

	async persistAdapterCredentials(
		organizationId: string,
		adapterType: ExportAdapterType,
		settings: Record<string, unknown>,
	) {
		await this.saveAdapterCredentials(organizationId, adapterType, settings)
	}

	private async mergeSavedCredentials(
		organizationId: string,
		adapterType: ExportAdapterType,
		settings: Record<string, unknown>,
		saveCredentials: boolean,
	): Promise<Record<string, unknown>> {
		const credentialKeys = this.getCredentialKeys(adapterType)
		if (credentialKeys.length === 0) return settings

		const saved = await this.getSavedAdapterCredentials(organizationId, adapterType)
		const merged = { ...(saved ?? {}), ...this.removeEmptyValues(settings) }

		if (adapterType === ExportAdapterType.HUGGINGFACE && !this.hasValue(merged.token)) {
			const orgToken = await this.getOrganizationHuggingFaceToken(organizationId)
			if (orgToken) merged.token = orgToken
		}

		const missingCredentialKeys = credentialKeys.filter(key => !this.hasValue(merged[key]))
		if (missingCredentialKeys.length > 0) {
			throw new BadRequestException(
				`Missing credentials for ${adapterType}: ${missingCredentialKeys.join(', ')}. Provide them in the export target, enable "Save credentials", or configure them in Settings.`,
			)
		}

		return merged
	}

	private async getSavedAdapterCredentials(
		organizationId: string,
		adapterType: ExportAdapterType,
	): Promise<Record<string, unknown> | null> {
		const credential = await this.prisma.exportCredential.findUnique({
			where: {
				organizationId_adapterType: {
					organizationId,
					adapterType,
				},
			},
		})

		if (!credential) return null
		return this.decryptSettings(credential.encryptedSettings, credential.iv, credential.authTag)
	}

	private async saveAdapterCredentials(
		organizationId: string,
		adapterType: ExportAdapterType,
		settings: Record<string, unknown>,
	) {
		const credentialKeys = this.getCredentialKeys(adapterType)
		if (credentialKeys.length === 0) return

		const credentials = credentialKeys.reduce<Record<string, unknown>>((acc, key) => {
			if (this.hasValue(settings[key])) acc[key] = settings[key]
			return acc
		}, {})

		if (Object.keys(credentials).length === 0) return

		const encrypted = this.encryptSettings(credentials)
		await this.prisma.exportCredential.upsert({
			where: {
				organizationId_adapterType: {
					organizationId,
					adapterType,
				},
			},
			create: {
				organizationId,
				adapterType,
				...encrypted,
			},
			update: encrypted,
		})
	}

	private getCredentialKeys(adapterType: ExportAdapterType): string[] {
		switch (adapterType) {
			case ExportAdapterType.HUGGINGFACE:
				return ['token']
			case ExportAdapterType.POSTGRES:
				return ['host', 'port', 'username', 'password', 'database']
			case ExportAdapterType.MONGODB:
				return ['uri', 'database']
			default:
				return []
		}
	}

	private removeEmptyValues(settings: Record<string, unknown>) {
		return Object.fromEntries(Object.entries(settings).filter(([, value]) => this.hasValue(value)))
	}

	private hasValue(value: unknown): boolean {
		return value !== undefined && value !== null && String(value).trim().length > 0
	}

	private encryptSettings(settings: Record<string, unknown>) {
		const iv = randomBytes(12)
		const cipher = createCipheriv('aes-256-gcm', this.getCredentialEncryptionKey(), iv)
		const encrypted = Buffer.concat([
			cipher.update(JSON.stringify(settings), 'utf8'),
			cipher.final(),
		])

		return {
			encryptedSettings: encrypted.toString('base64'),
			iv: iv.toString('base64'),
			authTag: cipher.getAuthTag().toString('base64'),
		}
	}

	private decryptSettings(encryptedSettings: string, iv: string, authTag: string) {
		const decipher = createDecipheriv(
			'aes-256-gcm',
			this.getCredentialEncryptionKey(),
			Buffer.from(iv, 'base64'),
		)
		decipher.setAuthTag(Buffer.from(authTag, 'base64'))
		const decrypted = Buffer.concat([
			decipher.update(Buffer.from(encryptedSettings, 'base64')),
			decipher.final(),
		])

		return JSON.parse(decrypted.toString('utf8')) as Record<string, unknown>
	}

	private getCredentialEncryptionKey() {
		return createHash('sha256').update(this.config.security.exportCredentialsSecret).digest()
	}

	/**
	 * Record export operation in database
	 */
	private async recordExport(
		pipelineRunId: string,
		adapterType: ExportAdapterType,
		result: ExportResult,
	) {
		try {
			await this.prisma.pipelineExport.create({
				data: {
					pipelineRunId,
					adapterType,
					destination: result.destination,
					recordsExported: result.recordsExported,
					metadata: JSON.parse(JSON.stringify(result.metadata ?? {})),
				},
			})
		} catch (error) {
			// Log but don't fail the export if recording fails
			this.logger.warn(
				`Failed to record export: ${error instanceof Error ? error.message : 'Unknown error'}`,
			)
		}
	}

	/**
	 * Get export history for a pipeline run
	 */
	async getExportHistory(pipelineRunId: string, organizationId: string) {
		const run = await this.prisma.pipelineRun.findUnique({
			where: { id: pipelineRunId },
		})

		if (!run || run.organizationId !== organizationId) {
			throw new NotFoundException('Pipeline run not found')
		}

		return this.prisma.pipelineExport.findMany({
			where: { pipelineRunId },
			orderBy: { createdAt: 'desc' },
		})
	}

	async getOrganizationHuggingFaceToken(organizationId: string): Promise<string | null> {
		const saved = await this.getSavedAdapterCredentials(organizationId, ExportAdapterType.HUGGINGFACE)
		const savedToken = saved?.token
		if (typeof savedToken === 'string' && savedToken.trim().length > 0) {
			return savedToken.trim()
		}
		return this.config.huggingFace.token ?? null
	}

	async saveOrganizationHuggingFaceToken(organizationId: string, token: string) {
		const normalized = token.trim()
		if (!normalized) {
			throw new BadRequestException('Hugging Face token cannot be empty')
		}
		await this.saveAdapterCredentials(organizationId, ExportAdapterType.HUGGINGFACE, { token: normalized })
	}

	async removeOrganizationHuggingFaceToken(organizationId: string) {
		await this.prisma.exportCredential.deleteMany({
			where: { organizationId, adapterType: ExportAdapterType.HUGGINGFACE },
		})
	}

	async getOrganizationHuggingFaceTokenPreview(organizationId: string) {
		const token = await this.getOrganizationHuggingFaceToken(organizationId)
		if (!token) return null
		if (token.length <= 8) return '••••••••'
		return `${token.slice(0, 4)}••••${token.slice(-4)}`
	}
}
