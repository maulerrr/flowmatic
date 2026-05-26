import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common'
import { whoAmI } from '@huggingface/hub'
import { AppConfigService } from 'src/common/config/config.service'
import { ExportService } from '../export/export.service'

export interface HuggingFaceIntegrationStatus {
	configured: boolean
	username?: string
	tokenPreview?: string
}

export interface HuggingFaceModelSummary {
	id: string
	author: string
	modelId: string
	private: boolean
	downloads: number
	likes: number
	tags: string[]
	pipelineTag?: string
	library?: string
	lastModified?: string
	resources: HuggingFaceModelResources
}

export interface HuggingFaceModelResources {
	recommendedDevice: 'cpu' | 'gpu'
	estimatedRamGb: number
	estimatedVramGb: number | null
	estimatedStorageGb: number
	inferenceProvider: 'flowmatic-local' | 'dedicated-gpu-recommended'
	notes: string[]
}

interface HuggingFaceApiModel {
	id: string
	modelId?: string
	author?: string
	private?: boolean
	downloads?: number
	likes?: number
	tags?: string[]
	pipeline_tag?: string
	library_name?: string
	lastModified?: string
	safetensors?: { total?: number; parameters?: Record<string, number> }
	siblings?: Array<{ rfilename?: string; size?: number }>
}

@Injectable()
export class HuggingFaceIntegrationService {
	constructor(
		private readonly exportService: ExportService,
		private readonly config: AppConfigService,
	) {}

	async getStatus(organizationId: string): Promise<HuggingFaceIntegrationStatus> {
		const token =
			(await this.exportService.getOrganizationHuggingFaceToken(organizationId)) ??
			this.config.huggingFace.token ??
			null
		if (!token) return { configured: false }

		try {
			const profile = await whoAmI({ accessToken: token })
			const tokenPreview =
				(await this.exportService.getOrganizationHuggingFaceTokenPreview(organizationId)) ?? undefined
			return {
				configured: true,
				username: profile.name,
				tokenPreview,
			}
		} catch {
			const tokenPreview =
				(await this.exportService.getOrganizationHuggingFaceTokenPreview(organizationId)) ?? undefined
			return {
				configured: true,
				tokenPreview,
			}
		}
	}

	async saveToken(organizationId: string, token: string) {
		await this.exportService.saveOrganizationHuggingFaceToken(organizationId, token)
		return this.getStatus(organizationId)
	}

	async removeToken(organizationId: string) {
		await this.exportService.removeOrganizationHuggingFaceToken(organizationId)
		return { configured: false }
	}

	async requireToken(organizationId: string) {
		const token = await this.exportService.getOrganizationHuggingFaceToken(organizationId)
		if (!token) {
			throw new BadRequestException(
				'Add a Hugging Face token in Settings or during signup to use Hub models.',
			)
		}
		return token
	}

	async listModels(
		organizationId: string,
		input: { search?: string; page?: number; limit?: number },
	) {
		const token = await this.requireToken(organizationId)
		const profile = await whoAmI({ accessToken: token })
		const page = input.page ?? 1
		const limit = input.limit ?? 12
		const offset = (page - 1) * limit
		const params = new URLSearchParams({
			author: profile.name,
			limit: String(limit),
			offset: String(offset),
			sort: 'lastModified',
			direction: '-1',
		})
		if (input.search?.trim()) params.set('search', input.search.trim())

		const response = await fetch(`https://huggingface.co/api/models?${params.toString()}`, {
			headers: { Authorization: `Bearer ${token}` },
		})
		if (!response.ok) {
			throw new BadRequestException(`Could not list Hugging Face models (${response.status})`)
		}

		const models = (await response.json()) as HuggingFaceApiModel[]
		const totalHeader = response.headers.get('X-Total-Count')
		const total = totalHeader ? Number(totalHeader) : models.length + offset

		return {
			username: profile.name,
			items: models.map(model => this.toSummary(model)),
			page,
			limit,
			total: Number.isFinite(total) ? total : models.length + offset,
			hasMore: models.length === limit,
		}
	}

	async getModelDetails(organizationId: string, modelId: string) {
		const token = await this.requireToken(organizationId)
		const normalized = modelId.replace(/^hf:/, '').trim()
		const response = await fetch(this.hfModelApiUrl(normalized), {
			headers: { Authorization: `Bearer ${token}` },
		})
		if (response.status === 404) throw new NotFoundException('Hugging Face model not found')
		if (!response.ok) {
			throw new BadRequestException(`Could not load model details (${response.status})`)
		}
		const model = (await response.json()) as HuggingFaceApiModel
		return this.toSummary(model)
	}

	async runInference(organizationId: string, modelId: string, payload: Record<string, unknown>) {
		const token = await this.requireToken(organizationId)
		const normalized = modelId.replace(/^hf:/, '').trim()
		return this.callInferenceService({
			modelId: normalized,
			token,
			inputs: payload,
		})
	}

	async runLocalInference(
		organizationId: string,
		localRun: string,
		payload: Record<string, unknown>,
	) {
		return this.callInferenceService({
			localRun,
			inputs: payload,
		})
	}

	async prefetchModel(
		organizationId: string,
		input: { modelId?: string; localRun?: string },
	) {
		const normalized = input.modelId?.replace(/^hf:/, '').trim()
		const token = normalized ? await this.requireToken(organizationId) : undefined
		const baseUrl = this.config.modelInference.baseUrl.replace(/\/$/, '')
		let response: Response
		try {
			response = await fetch(`${baseUrl}/v1/prefetch`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					modelId: normalized,
					localRun: input.localRun,
					token,
				}),
			})
		} catch {
			throw new BadRequestException(
				`Cannot reach model inference service at ${baseUrl}. Ensure model-inference is running (MODEL_INFERENCE_URL).`,
			)
		}
		const body = await response.json().catch(() => ({}))
		if (!response.ok) {
			throw new BadRequestException(this.extractInferenceError(body, 'Model prefetch failed'))
		}
		return body
	}

	private async callInferenceService(body: {
		modelId?: string
		localRun?: string
		token?: string
		inputs: Record<string, unknown>
	}) {
		const baseUrl = this.config.modelInference.baseUrl.replace(/\/$/, '')
		let response: Response
		try {
			response = await fetch(`${baseUrl}/v1/infer`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
			})
		} catch {
			throw new BadRequestException(
				`Cannot reach model inference service at ${baseUrl}. Ensure model-inference is running (MODEL_INFERENCE_URL).`,
			)
		}
		const responseBody = await response.json().catch(() => ({}))
		if (!response.ok) {
			throw new BadRequestException(
				this.extractInferenceError(responseBody, 'Model inference request failed'),
			)
		}
		return responseBody
	}

	private extractInferenceError(body: unknown, fallback: string): string {
		if (typeof body === 'object' && body !== null) {
			const record = body as Record<string, unknown>
			if (typeof record.detail === 'string') return record.detail
			if (typeof record.error === 'string') return record.error
		}
		return fallback
	}

	private hfModelApiUrl(modelId: string): string {
		const slash = modelId.indexOf('/')
		if (slash <= 0) {
			throw new BadRequestException('Model ID must use the format username/model-name')
		}
		const author = modelId.slice(0, slash)
		const repo = modelId.slice(slash + 1)
		return `https://huggingface.co/api/models/${encodeURIComponent(author)}/${encodeURIComponent(repo)}`
	}

	private toSummary(model: HuggingFaceApiModel): HuggingFaceModelSummary {
		const id = model.id ?? model.modelId ?? ''
		const [author, ...rest] = id.split('/')
		return {
			id,
			author: model.author ?? author ?? 'unknown',
			modelId: id,
			private: Boolean(model.private),
			downloads: model.downloads ?? 0,
			likes: model.likes ?? 0,
			tags: model.tags ?? [],
			pipelineTag: model.pipeline_tag,
			library: model.library_name,
			lastModified: model.lastModified,
			resources: this.estimateResources(model),
		}
	}

	private estimateResources(model: HuggingFaceApiModel): HuggingFaceModelResources {
		const siblingBytes =
			model.siblings?.reduce((sum, item) => sum + (item.size ?? 0), 0) ?? 0
		const totalBytes = model.safetensors?.total ?? siblingBytes
		const storageGb = Math.max(0.01, totalBytes / 1_000_000_000)
		const id = model.id ?? model.modelId ?? ''
		const isFlowmatic = id.includes('flowmatic-')
		const pipelineTag = model.pipeline_tag ?? ''
		const gpuHeavy = !isFlowmatic && storageGb >= 2
		const notes: string[] = []
		if (model.private) notes.push('Private model — token must have read access.')
		if (isFlowmatic) {
			notes.push('Flowmatic checkpoint — downloaded once and run locally on CPU/GPU.')
			notes.push(`Estimated checkpoint size: ~${storageGb.toFixed(2)} GB.`)
		} else if (storageGb >= 10) {
			notes.push('Large checkpoint — dedicated GPU recommended for self-hosting.')
		}
		if (pipelineTag) notes.push(`Primary task: ${pipelineTag}.`)

		return {
			recommendedDevice: isFlowmatic || gpuHeavy ? 'gpu' : 'cpu',
			estimatedRamGb: Math.max(1, Math.ceil(storageGb * 1.4)),
			estimatedVramGb: isFlowmatic ? Math.max(1, Math.ceil(storageGb * 1.2)) : gpuHeavy ? Math.max(4, Math.ceil(storageGb * 1.1)) : null,
			estimatedStorageGb: Number(storageGb.toFixed(2)),
			inferenceProvider: isFlowmatic ? 'flowmatic-local' : 'dedicated-gpu-recommended',
			notes,
		}
	}
}
