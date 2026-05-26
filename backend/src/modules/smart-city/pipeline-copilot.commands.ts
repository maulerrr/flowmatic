import { CopilotVizCommand } from './pipeline-copilot.types'

export const VIZ_COMMAND_CATALOG: CopilotVizCommand[] = [
	{
		id: 'event_volume',
		slash: 'event-volume',
		label: 'Event volume',
		description: 'Sensor events per UTC hour (24h area chart)',
		category: 'data',
	},
	{
		id: 'export_health',
		slash: 'export-health',
		label: 'Export health',
		description: 'Succeeded vs failed export runs (donut)',
		category: 'exports',
	},
	{
		id: 'data_funnel',
		slash: 'funnel',
		label: 'Data funnel',
		description: 'Raw → cleaned → business stage counts',
		category: 'data',
	},
	{
		id: 'sources_status',
		slash: 'sources',
		label: 'Source health',
		description: 'Status breakdown per ingest source',
		category: 'sources',
	},
	{
		id: 'export_adapters',
		slash: 'adapters',
		label: 'By adapter',
		description: 'Export volume grouped by adapter',
		category: 'exports',
	},
	{
		id: 'recent_exports',
		slash: 'exports',
		label: 'Recent exports',
		description: 'Latest export run table',
		category: 'exports',
	},
	{
		id: 'core_unit',
		slash: 'core',
		label: 'Core unit',
		description: 'Active model and processing flags',
		category: 'core',
	},
	{
		id: 'data_quality',
		slash: 'quality',
		label: 'Data quality',
		description: 'Missing, duplicate, and outlier sample KPIs',
		category: 'data',
	},
	{
		id: 'export_failures',
		slash: 'export-failures',
		label: 'Export failures',
		description: 'Failed export run outcomes',
		category: 'exports',
	},
	{
		id: 'source_errors',
		slash: 'source-errors',
		label: 'Source errors',
		description: 'Sources reporting errors',
		category: 'sources',
	},
	{
		id: 'pipeline_overview',
		slash: 'overview',
		label: 'Pipeline overview',
		description: 'Medallion funnel snapshot',
		category: 'data',
	},
]

const SLASH_ALIASES: Record<string, string> = {
	'event-volume': 'event_volume',
	events: 'event_volume',
	'export-health': 'export_health',
	funnel: 'data_funnel',
	'data-funnel': 'data_funnel',
	sources: 'sources_status',
	'source-health': 'sources_status',
	adapters: 'export_adapters',
	exports: 'recent_exports',
	'recent-exports': 'recent_exports',
	core: 'core_unit',
	quality: 'data_quality',
	overview: 'pipeline_overview',
	'export-failures': 'export_failures',
	'source-errors': 'source_errors',
}

export function resolveSlashCommand(token: string): string | null {
	const normalized = token.toLowerCase().trim().replace(/_/g, '-')
	return SLASH_ALIASES[normalized] ?? null
}

export function parseComposerMessage(message: string): {
	questionText: string
	vizChipIds: string[]
} {
	const slashRegex = /\/([a-z0-9_-]+)/gi
	const vizChipIds: string[] = []
	for (const match of message.matchAll(slashRegex)) {
		const chipId = resolveSlashCommand(match[1])
		if (chipId && !vizChipIds.includes(chipId)) vizChipIds.push(chipId)
	}
	const questionText = message.replace(slashRegex, ' ').replace(/\s+/g, ' ').trim()
	return { questionText, vizChipIds }
}

export function buildVizCommandsForChips(chipIds: string[]): CopilotVizCommand[] {
	const allowed = new Set(chipIds)
	return VIZ_COMMAND_CATALOG.filter(command => allowed.has(command.id))
}
