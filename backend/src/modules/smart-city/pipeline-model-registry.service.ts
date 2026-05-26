import { Injectable } from '@nestjs/common'
import { existsSync, readFileSync, readdirSync } from 'fs'
import { resolve } from 'path'
import { ModelModality, ModelRegistryEntry, ModelTask } from './pipeline-model-router.types'

const KIND_PROFILE: Record<
	string,
	{
		modality: ModelModality
		tasks: ModelTask[]
		sensorKinds: string[]
		requiresGeo?: boolean
		priority: number
	}
> = {
	tranad_anomaly: {
		modality: 'traffic',
		tasks: ['anomaly'],
		sensorKinds: ['traffic', 'generic'],
		priority: 1,
	},
	autoencoder: {
		modality: 'traffic',
		tasks: ['anomaly', 'repair'],
		sensorKinds: ['traffic'],
		priority: 2,
	},
	stgcn_forecast: {
		modality: 'traffic',
		tasks: ['forecast'],
		sensorKinds: ['traffic'],
		requiresGeo: true,
		priority: 2,
	},
	patchtst_forecast: {
		modality: 'traffic',
		tasks: ['forecast'],
		sensorKinds: ['traffic'],
		priority: 3,
	},
	gru_forecast: {
		modality: 'traffic',
		tasks: ['forecast'],
		sensorKinds: ['traffic'],
		priority: 4,
	},
	itransformer_forecast: {
		modality: 'traffic',
		tasks: ['forecast'],
		sensorKinds: ['traffic'],
		priority: 5,
	},
	tcn_forecast: {
		modality: 'traffic',
		tasks: ['forecast'],
		sensorKinds: ['traffic'],
		priority: 6,
	},
	transformer_classifier: {
		modality: 'traffic',
		tasks: ['classification'],
		sensorKinds: ['traffic'],
		priority: 7,
	},
	saits_imputer: {
		modality: 'generic',
		tasks: ['imputation'],
		sensorKinds: ['traffic', 'weather', 'air', 'generic'],
		priority: 8,
	},
	timesblock_forecast: {
		modality: 'weather',
		tasks: ['forecast'],
		sensorKinds: ['weather', 'air'],
		priority: 1,
	},
	transformer_forecast: {
		modality: 'weather',
		tasks: ['forecast'],
		sensorKinds: ['weather', 'air'],
		priority: 2,
	},
	dlinear_forecast: {
		modality: 'energy',
		tasks: ['forecast'],
		sensorKinds: ['generic'],
		priority: 20,
	},
	nlinear_forecast: {
		modality: 'energy',
		tasks: ['forecast'],
		sensorKinds: ['generic'],
		priority: 21,
	},
}

type ProductionPortfolio = {
	models?: Array<{
		run: string
		slot: string
		kind: string
		dataset: string
		priority: number
		capabilities?: {
			modality?: ModelModality
			tasks?: ModelTask[]
			sensorKinds?: string[]
			requiresGeo?: boolean
		}
	}>
}

function resolveWorkspacePath(...parts: string[]) {
	const candidates = [resolve(process.cwd(), ...parts), resolve(process.cwd(), '..', ...parts)]
	for (const candidate of candidates) {
		if (existsSync(candidate)) return candidate
	}
	return candidates[0]
}

function inferModalityFromDataset(dataset: string): ModelModality {
	if (dataset.includes('weather')) return 'weather'
	if (dataset.includes('ett') || dataset.includes('energy')) return 'energy'
	if (dataset.includes('traffic') || dataset.includes('astana') || dataset.includes('pems') || dataset.includes('metr'))
		return 'traffic'
	return 'generic'
}

function loadProductionPortfolio(): ProductionPortfolio | null {
	const portfolioPath = resolveWorkspacePath('models', 'reports', 'production_portfolio.json')
	if (!existsSync(portfolioPath)) return null
	return JSON.parse(readFileSync(portfolioPath, 'utf8')) as ProductionPortfolio
}

function buildEntry(input: {
	id: string
	label: string
	kind: string
	dataset: string
	source: ModelRegistryEntry['source']
	priority?: number
	modality?: ModelModality
	tasks?: ModelTask[]
	sensorKinds?: string[]
	requiresGeo?: boolean
	production?: boolean
	productionSlot?: string
	run?: string
	repoId?: string
}): ModelRegistryEntry | null {
	const profile = KIND_PROFILE[input.kind]
	if (!profile && !input.modality) return null
	return {
		id: input.id,
		label: input.label,
		kind: input.kind,
		dataset: input.dataset,
		modality: input.modality ?? profile?.modality ?? inferModalityFromDataset(input.dataset),
		tasks: input.tasks ?? profile?.tasks ?? ['forecast'],
		sensorKinds: input.sensorKinds ?? profile?.sensorKinds ?? ['generic'],
		requiresGeo: input.requiresGeo ?? profile?.requiresGeo,
		priority: input.priority ?? profile?.priority ?? 50,
		source: input.source,
		production: input.production,
		productionSlot: input.productionSlot,
		run: input.run,
		repoId: input.repoId,
	}
}

@Injectable()
export class PipelineModelRegistryService {
	private cache: ModelRegistryEntry[] | null = null

	listModels(force = false): ModelRegistryEntry[] {
		if (this.cache && !force) return this.cache
		const portfolio = loadProductionPortfolio()
		const portfolioRuns = new Set((portfolio?.models ?? []).map(item => item.run))
		const portfolioByRun = new Map((portfolio?.models ?? []).map(item => [item.run, item]))
		const entries = new Map<string, ModelRegistryEntry>()

		if (portfolio && portfolio.models?.length) {
			const checkpointsDir = resolveWorkspacePath('models', 'checkpoints')
			for (const item of portfolio.models) {
				const metadataPath = resolve(checkpointsDir, item.run, 'metadata.json')
				if (!existsSync(metadataPath)) continue
				const metadata = JSON.parse(readFileSync(metadataPath, 'utf8')) as Record<string, unknown>
				const capabilities = (metadata.capabilities ?? item.capabilities ?? {}) as Record<string, unknown>
				const entry = buildEntry({
					id: `production:${item.run}`,
					label: item.run,
					kind: String(metadata.kind ?? item.kind),
					dataset: String(metadata.dataset ?? item.dataset),
					source: 'production',
					priority: item.priority,
					modality: capabilities.modality as ModelModality | undefined,
					tasks: capabilities.tasks as ModelTask[] | undefined,
					sensorKinds: capabilities.sensorKinds as string[] | undefined,
					requiresGeo: Boolean(capabilities.requiresGeo),
					production: true,
					productionSlot: item.slot,
					run: item.run,
				})
				if (entry) entries.set(entry.id, entry)
			}
		}

		const manifestPath = resolveWorkspacePath('models', 'reports', 'huggingface_model_manifest.json')
		if (existsSync(manifestPath)) {
			const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
				models?: Array<{
					repoId: string
					sourceRun: string
					kind: string
					dataset: string
					slug?: string
				}>
			}
			for (const model of manifest.models ?? []) {
				if (portfolioRuns.size > 0 && !portfolioRuns.has(model.sourceRun)) continue
				const portfolioItem = portfolioByRun.get(model.sourceRun)
				const entry = buildEntry({
					id: portfolioItem ? `production:${model.sourceRun}` : `research:${model.sourceRun}`,
					label: model.slug ?? model.repoId,
					kind: model.kind,
					dataset: model.dataset,
					source: portfolioItem ? 'production' : 'manifest',
					priority: portfolioItem?.priority,
					production: Boolean(portfolioItem),
					productionSlot: portfolioItem?.slot,
					run: model.sourceRun,
					repoId: model.repoId,
				})
				if (entry) entries.set(entry.id, entry)
			}
		}

		const checkpointsDir = resolveWorkspacePath('models', 'checkpoints')
		if (existsSync(checkpointsDir)) {
			for (const dir of readdirSync(checkpointsDir, { withFileTypes: true })) {
				if (!dir.isDirectory()) continue
				if (portfolioRuns.size > 0 && !portfolioRuns.has(dir.name)) continue
				const metadataPath = resolve(checkpointsDir, dir.name, 'metadata.json')
				if (!existsSync(metadataPath)) continue
				const metadata = JSON.parse(readFileSync(metadataPath, 'utf8')) as Record<string, unknown>
				const kind = String(metadata.kind ?? '')
				const dataset = String(metadata.dataset ?? 'generic')
				const portfolioItem = portfolioByRun.get(dir.name)
				const capabilities = (metadata.capabilities ?? {}) as Record<string, unknown>
				const entry = buildEntry({
					id: portfolioItem ? `production:${dir.name}` : `research:${dir.name}`,
					label: dir.name,
					kind,
					dataset,
					source: portfolioItem ? 'production' : 'research',
					priority: portfolioItem?.priority ?? (metadata.production as { priority?: number } | undefined)?.priority,
					modality: capabilities.modality as ModelModality | undefined,
					tasks: capabilities.tasks as ModelTask[] | undefined,
					sensorKinds: capabilities.sensorKinds as string[] | undefined,
					requiresGeo: Boolean(capabilities.requiresGeo),
					production: Boolean(portfolioItem),
					productionSlot: portfolioItem?.slot,
					run: dir.name,
				})
				if (entry) entries.set(entry.id, entry)
			}
		}

		this.cache = [...entries.values()].sort((left, right) => {
			if (left.priority !== right.priority) return left.priority - right.priority
			return left.label.localeCompare(right.label)
		})
		return this.cache
	}

	findById(modelId: string): ModelRegistryEntry | undefined {
		return this.listModels().find(entry => entry.id === modelId)
	}
}
