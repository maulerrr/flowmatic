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
	if (dataset.includes('traffic') || dataset.includes('astana')) return 'traffic'
	return 'generic'
}

function buildEntry(input: {
	id: string
	label: string
	kind: string
	dataset: string
	source: 'research' | 'manifest'
	run?: string
	repoId?: string
}): ModelRegistryEntry | null {
	const profile = KIND_PROFILE[input.kind]
	if (!profile) return null
	return {
		id: input.id,
		label: input.label,
		kind: input.kind,
		dataset: input.dataset,
		modality: profile.modality ?? inferModalityFromDataset(input.dataset),
		tasks: profile.tasks,
		sensorKinds: profile.sensorKinds,
		requiresGeo: profile.requiresGeo,
		priority: profile.priority,
		source: input.source,
		run: input.run,
		repoId: input.repoId,
	}
}

@Injectable()
export class PipelineModelRegistryService {
	private cache: ModelRegistryEntry[] | null = null

	listModels(force = false): ModelRegistryEntry[] {
		if (this.cache && !force) return this.cache
		const entries = new Map<string, ModelRegistryEntry>()

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
				const entry = buildEntry({
					id: `research:${model.sourceRun}`,
					label: model.slug ?? model.repoId,
					kind: model.kind,
					dataset: model.dataset,
					source: 'manifest',
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
				const metadataPath = resolve(checkpointsDir, dir.name, 'metadata.json')
				if (!existsSync(metadataPath)) continue
				const metadata = JSON.parse(readFileSync(metadataPath, 'utf8')) as Record<string, unknown>
				const kind = String(metadata.kind ?? '')
				const dataset = String(metadata.dataset ?? 'generic')
				const entry = buildEntry({
					id: `research:${dir.name}`,
					label: dir.name,
					kind,
					dataset,
					source: 'research',
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
