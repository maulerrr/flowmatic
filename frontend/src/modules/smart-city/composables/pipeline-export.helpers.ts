import type { ExportAdapterKind, ExportStage } from './pipeline-workbench.types'

export type { ExportAdapterKind, ExportStage } from './pipeline-workbench.types'

export interface ExportTargetFormState {
	stage: ExportStage
	adapterType: ExportAdapterKind
	saveCredentials: boolean
	isContinuous: boolean
}

export interface ExportSettingsFormState {
	jsonPrettyPrint: boolean
	csvDelimiter: string
	postgresHost: string
	postgresPort: number
	postgresUsername: string
	postgresPassword: string
	postgresDatabase: string
	postgresTable: string
	postgresIfExists: 'append' | 'replace'
	mongodbUri: string
	mongodbDatabase: string
	mongodbCollection: string
	mongodbIfExists: 'append' | 'replace'
	huggingFaceToken: string
	huggingFaceRepoName: string
	huggingFaceCommitMessage: string
	huggingFacePrivate: boolean
	advancedJson: string
}

export interface ExportValidationContext {
	form: ExportTargetFormState
	settings: ExportSettingsFormState
	usesSavedHuggingFaceToken: boolean
	isEditing: boolean
}

export function applyAdapterDefaults(
	form: ExportTargetFormState,
	settings: ExportSettingsFormState,
): ExportSettingsFormState {
	const next = { ...settings }
	switch (form.adapterType) {
		case 'json':
			next.jsonPrettyPrint = true
			break
		case 'csv':
			if (!next.csvDelimiter) next.csvDelimiter = ','
			break
		case 'postgres':
			if (!next.postgresPort) next.postgresPort = 5432
			if (!next.postgresTable) next.postgresTable = `smart_city_${form.stage}`
			break
		case 'mongodb':
			if (!next.mongodbCollection) next.mongodbCollection = `smart_city_${form.stage}`
			break
		case 'huggingface':
			break
	}
	return next
}

export function syncStageDependentExportFields(
	stage: ExportStage,
	settings: ExportSettingsFormState,
): ExportSettingsFormState {
	const next = { ...settings }
	const tableName = `smart_city_${stage}`
	if (!next.postgresTable || next.postgresTable.startsWith('smart_city_')) {
		next.postgresTable = tableName
	}
	if (!next.mongodbCollection || next.mongodbCollection.startsWith('smart_city_')) {
		next.mongodbCollection = tableName
	}
	return next
}

export function buildStructuredExportSettings(
	form: ExportTargetFormState,
	settings: ExportSettingsFormState,
): Record<string, unknown> {
	switch (form.adapterType) {
		case 'json':
			return {
				prettyPrint: settings.jsonPrettyPrint,
				ifExists: form.isContinuous ? 'append' : 'replace',
			}
		case 'csv':
			return {
				delimiter: settings.csvDelimiter || ',',
			}
		case 'postgres': {
			const payload: Record<string, unknown> = {
				host: settings.postgresHost.trim(),
				port: Number(settings.postgresPort),
				username: settings.postgresUsername.trim(),
				database: settings.postgresDatabase.trim(),
				table: settings.postgresTable.trim(),
				ifExists: settings.postgresIfExists,
			}
			if (settings.postgresPassword.trim()) payload.password = settings.postgresPassword
			return payload
		}
		case 'mongodb':
			return {
				uri: settings.mongodbUri.trim(),
				database: settings.mongodbDatabase.trim(),
				collection: settings.mongodbCollection.trim(),
				ifExists: settings.mongodbIfExists,
			}
		case 'huggingface': {
			const payload: Record<string, unknown> = {
				repoName: settings.huggingFaceRepoName.trim(),
				commitMessage: settings.huggingFaceCommitMessage.trim(),
				private: settings.huggingFacePrivate,
				ifExists: form.isContinuous ? 'append' : 'replace',
			}
			if (settings.huggingFaceToken.trim()) payload.token = settings.huggingFaceToken.trim()
			return payload
		}
		default:
			return {}
	}
}

export function parseAdvancedExportSettings(
	advancedJson: string,
	onInvalid?: () => void,
): Record<string, unknown> {
	if (!advancedJson.trim()) return {}
	try {
		return JSON.parse(advancedJson) as Record<string, unknown>
	} catch {
		onInvalid?.()
		return {}
	}
}

export function buildExportSettings(
	form: ExportTargetFormState,
	settings: ExportSettingsFormState,
	onInvalidAdvanced?: () => void,
): Record<string, unknown> {
	const structured = buildStructuredExportSettings(form, settings)
	const advanced = parseAdvancedExportSettings(settings.advancedJson, onInvalidAdvanced)
	return { ...structured, ...advanced }
}

export function validateExportTargetForm(context: ExportValidationContext): string | null {
	const { form, settings, usesSavedHuggingFaceToken, isEditing } = context
	switch (form.adapterType) {
		case 'postgres':
			if (!settings.postgresHost.trim()) return 'PostgreSQL host is required'
			if (!settings.postgresDatabase.trim()) return 'PostgreSQL database is required'
			if (!settings.postgresTable.trim()) return 'PostgreSQL table is required'
			if (!settings.postgresPassword.trim() && !form.saveCredentials && !isEditing) {
				return 'PostgreSQL password is required (or enable Save credentials after entering it once)'
			}
			return null
		case 'mongodb':
			if (!settings.mongodbUri.trim()) return 'MongoDB URI is required'
			if (!settings.mongodbDatabase.trim()) return 'MongoDB database is required'
			if (!settings.mongodbCollection.trim()) return 'MongoDB collection is required'
			return null
		case 'huggingface':
			if (!settings.huggingFaceRepoName.trim()) return 'Hugging Face dataset repo name is required'
			if (!usesSavedHuggingFaceToken && !settings.huggingFaceToken.trim() && !form.saveCredentials) {
				return 'Hugging Face token is required (or save one in Settings)'
			}
			return null
		default:
			return null
	}
}

export function applyExportSettingsToForm(
	adapterType: ExportAdapterKind,
	stage: ExportStage,
	settings: Record<string, unknown>,
	current: ExportSettingsFormState,
): ExportSettingsFormState {
	const next: ExportSettingsFormState = { ...current, advancedJson: '' }
	switch (adapterType) {
		case 'json':
			next.jsonPrettyPrint = Boolean(settings.prettyPrint ?? true)
			break
		case 'csv':
			next.csvDelimiter = String(settings.delimiter ?? ',')
			break
		case 'postgres':
			next.postgresHost = String(settings.host ?? '')
			next.postgresPort = Number(settings.port ?? 5432)
			next.postgresUsername = String(settings.username ?? '')
			next.postgresPassword = String(settings.password ?? '')
			next.postgresDatabase = String(settings.database ?? '')
			next.postgresTable = String(settings.table ?? `smart_city_${stage}`)
			next.postgresIfExists = settings.ifExists === 'replace' ? 'replace' : 'append'
			break
		case 'mongodb':
			next.mongodbUri = String(settings.uri ?? '')
			next.mongodbDatabase = String(settings.database ?? '')
			next.mongodbCollection = String(settings.collection ?? `smart_city_${stage}`)
			next.mongodbIfExists = settings.ifExists === 'replace' ? 'replace' : 'append'
			break
		case 'huggingface':
			next.huggingFaceToken = String(settings.token ?? '')
			next.huggingFaceRepoName = String(settings.repoName ?? '')
			next.huggingFaceCommitMessage = String(settings.commitMessage ?? '')
			next.huggingFacePrivate = Boolean(settings.private ?? false)
			break
	}
	return next
}

export function buildExportSettingsPreview(
	adapterType: ExportAdapterKind,
	settings: ExportSettingsFormState,
): string {
	switch (adapterType) {
		case 'json':
			return settings.jsonPrettyPrint ? 'Pretty JSON file in object storage' : 'Compact JSON file in object storage'
		case 'csv':
			return `CSV file with "${settings.csvDelimiter}" delimiter`
		case 'postgres':
			return `${settings.postgresHost || 'host'} / ${settings.postgresDatabase || 'database'} / ${settings.postgresTable || 'table'}`
		case 'mongodb':
			return `${settings.mongodbDatabase || 'database'} / ${settings.mongodbCollection || 'collection'}`
		case 'huggingface':
			return settings.huggingFaceRepoName
				? `HF ${settings.huggingFaceRepoName} · hourly CSV parts in data/hourly/`
				: 'Hourly UTC CSV parts on Hugging Face'
		default:
			return 'Configure adapter settings'
	}
}

export function buildPipelineFlowHint(input: {
	isLive: boolean
	runningSourceCount: number
	recentEventCount: number
	continuousExportTargets: Array<{ name?: string | null; adapterType: string }>
	exportErrorCount: number
}): string {
	if (!input.isLive) {
		return 'Start the pipeline to stream events through sources → core → lake → exports.'
	}
	const parts = [
		`${input.runningSourceCount} source(s) feeding`,
		`${input.recentEventCount} recent events`,
	]
	if (input.continuousExportTargets.length > 0) {
		const count = input.continuousExportTargets.length
		const labels = input.continuousExportTargets
			.map(target => target.name?.trim() || target.adapterType)
			.slice(0, 3)
			.join(', ')
		const overflow = count > 3 ? ` +${count - 3} more` : ''
		parts.push(`${count} export stream${count === 1 ? '' : 's'} active (${labels}${overflow})`)
	}
	if (input.exportErrorCount > 0) {
		parts.push(`${input.exportErrorCount} export target(s) need attention`)
	}
	return parts.join(' · ')
}
