import { BaseExportAdapter } from './base.adapter'
import {
	ExportAdapterType,
	ExportConfig,
	ExportResult,
	HuggingFaceConfig,
} from '../types/export.types'
import { whoAmI, createRepo, uploadFile, downloadFile } from '@huggingface/hub'
import { parseCsvBuffer, rowsToCsv } from 'src/common/utils/csv-parser.util'
import {
	HOURLY_MANIFEST_PATH,
	HourlyCsvManifest,
	HourlyPartUploadSummary,
	buildDatasetDescription,
	buildHourlyReadme,
	createEmptyHourlyManifest,
	groupRowsByHour,
	hourlyPartPath,
	mergeRowsByEventId,
	updateManifestPart,
} from '../utils/hourly-csv-partition.util'

type DatasetRepo = { type: 'dataset'; name: string }

/**
 * Hugging Face Export Adapter
 * Uploads cleaned data to Hugging Face Datasets as hourly UTC CSV parts.
 */
export class HuggingFaceExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.HUGGINGFACE
	name = 'Hugging Face Datasets'
	description = 'Export data to Hugging Face Datasets hub as hourly UTC CSV parts'
	requiredSettings = ['token', 'repoName']

	async validate(
		settings: Record<string, unknown>,
	): Promise<{ valid: boolean; errors?: string[] }> {
		const baseValidation = await super.validate(settings)
		if (!baseValidation.valid) return baseValidation

		const errors: string[] = []
		const config = settings as unknown as HuggingFaceConfig

		if (!config.token || config.token.trim().length === 0) {
			errors.push('Hugging Face token cannot be empty')
		}

		if (!config.repoName || config.repoName.trim().length === 0) {
			errors.push('Repository name cannot be empty')
		}

		if (config.repoName && !/^[a-zA-Z0-9_-]+$/.test(config.repoName)) {
			errors.push(
				'Repository name can only contain alphanumeric characters, hyphens, and underscores',
			)
		}

		if (errors.length > 0) {
			return { valid: false, errors }
		}

		return { valid: true }
	}

	async export(data: Record<string, unknown>[], config: ExportConfig): Promise<ExportResult> {
		const hfConfig = config.settings as unknown as HuggingFaceConfig
		await this.validate(config.settings)

		try {
			const convertedData = this.convertData(data)
			const user = await whoAmI({ accessToken: hfConfig.token })
			const fullRepoId = `${user.name}/${hfConfig.repoName.trim()}`
			const repo = this.datasetRepo(fullRepoId)

			if (convertedData.length === 0) {
				return {
					success: true,
					adapterType: this.type,
					fileName: HOURLY_MANIFEST_PATH,
					destination: `https://huggingface.co/datasets/${fullRepoId}`,
					recordsExported: 0,
					message: 'No new rows to export',
				}
			}

			await this.ensureDatasetRepo(repo, hfConfig, fullRepoId)

			const hourlyGroups = groupRowsByHour(convertedData)
			let manifest = (await this.loadManifest(repo, hfConfig.token)) ?? createEmptyHourlyManifest()
			const uploadedParts: HourlyPartUploadSummary[] = []
			const previewRows: Record<string, unknown>[] = []

			for (const [hourKey, hourRows] of hourlyGroups.entries()) {
				const partPath = hourlyPartPath(hourKey)
				let exportRows = hourRows

				if ((hfConfig.ifExists ?? 'append') === 'append') {
					const existing = await this.loadExistingCsv(repo, partPath, hfConfig.token)
					exportRows = mergeRowsByEventId(existing, hourRows)
				}

				const csvContent = rowsToCsv(exportRows)
				await uploadFile({
					repo,
					file: {
						path: partPath,
						content: new Blob([csvContent], { type: 'text/csv' }),
					},
					commitTitle:
						hfConfig.commitMessage ||
						`Append ${hourRows.length} rows to ${hourKey} (${exportRows.length} in hour file)`,
					accessToken: hfConfig.token,
				})

				const summary: HourlyPartUploadSummary = {
					hourKey,
					path: partPath,
					newRows: hourRows.length,
					totalRows: exportRows.length,
				}
				uploadedParts.push(summary)
				manifest = updateManifestPart(manifest, summary)
				previewRows.push(...exportRows.slice(-5))
			}

			await uploadFile({
				repo,
				file: {
					path: HOURLY_MANIFEST_PATH,
					content: new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }),
				},
				commitTitle: `Update hourly manifest (${manifest.totalRows} total rows)`,
				accessToken: hfConfig.token,
			})

			const readmeContent = buildHourlyReadme({
				manifest,
				pipelineRunId: config.pipelineRunId,
				fullRepoId,
				previewRows,
			})
			await uploadFile({
				repo,
				file: {
					path: 'README.md',
					content: new Blob([readmeContent], { type: 'text/markdown' }),
				},
				commitTitle: `Update dataset README (${manifest.totalRows} rows across hourly parts)`,
				accessToken: hfConfig.token,
			})

			const destination = uploadedParts[0]?.path ?? HOURLY_MANIFEST_PATH
			this.logExport(
				config,
				convertedData.length,
				`https://huggingface.co/datasets/${fullRepoId}`,
				'success',
			)

			return {
				success: true,
				adapterType: this.type,
				fileName: destination,
				destination: `https://huggingface.co/datasets/${fullRepoId}/tree/main/data/hourly`,
				recordsExported: convertedData.length,
				message: `Exported ${convertedData.length} new rows across ${uploadedParts.length} hourly file(s) (${manifest.totalRows} total rows in dataset)`,
				metadata: {
					repoId: fullRepoId,
					partitionScheme: manifest.partitionScheme,
					manifestPath: HOURLY_MANIFEST_PATH,
					hourlyParts: uploadedParts,
					totalRowsInDataset: manifest.totalRows,
					datasetDescription: buildDatasetDescription(manifest),
					hubUrl: `https://huggingface.co/datasets/${fullRepoId}`,
				},
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, hfConfig.repoName, 'failed')
			throw new Error(`Hugging Face export failed: ${message}`)
		}
	}

	private datasetRepo(fullRepoId: string): DatasetRepo {
		return { type: 'dataset', name: fullRepoId }
	}

	private async ensureDatasetRepo(repo: DatasetRepo, hfConfig: HuggingFaceConfig, fullRepoId: string) {
		const description = buildDatasetDescription(createEmptyHourlyManifest())
		try {
			await createRepo({
				repo,
				private: hfConfig.private || false,
				accessToken: hfConfig.token,
				description,
			})
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			if (this.isExistingRepoError(message)) return
			throw new Error(`Failed to create Hugging Face dataset repo: ${message}`)
		}
	}

	private isExistingRepoError(message: string): boolean {
		return /already exists|already created|repo.*exist|duplicate|409/i.test(message)
	}

	private async loadExistingCsv(
		repo: DatasetRepo,
		fileName: string,
		token: string,
	): Promise<Record<string, unknown>[]> {
		try {
			const file = await downloadFile({
				repo,
				path: fileName,
				accessToken: token,
			})
			if (!file) return []
			const buffer = Buffer.from(await file.arrayBuffer())
			return parseCsvBuffer(buffer).rows
		} catch {
			return []
		}
	}

	private async loadManifest(repo: DatasetRepo, token: string): Promise<HourlyCsvManifest | null> {
		try {
			const file = await downloadFile({
				repo,
				path: HOURLY_MANIFEST_PATH,
				accessToken: token,
			})
			if (!file) return null
			const parsed = JSON.parse(await file.text()) as HourlyCsvManifest
			if (!parsed || typeof parsed !== 'object') return null
			return parsed
		} catch {
			return null
		}
	}
}
