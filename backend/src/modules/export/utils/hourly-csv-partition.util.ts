export const HOURLY_PARTITION_PREFIX = 'data/hourly'
export const HOURLY_MANIFEST_PATH = `${HOURLY_PARTITION_PREFIX}/manifest.json`

export interface HourlyCsvManifestPart {
	path: string
	rowCount: number
	hourKey: string
	updatedAt: string
}

export interface HourlyCsvManifest {
	partitionScheme: 'hourly-utc'
	prefix: string
	description: string
	parts: Record<string, HourlyCsvManifestPart>
	totalRows: number
	updatedAt: string
}

export interface HourlyPartUploadSummary {
	hourKey: string
	path: string
	newRows: number
	totalRows: number
}

export function createEmptyHourlyManifest(): HourlyCsvManifest {
	return {
		partitionScheme: 'hourly-utc',
		prefix: `${HOURLY_PARTITION_PREFIX}/`,
		description:
			'Flowmatic smart-city export stored as hourly UTC CSV parts. Each file covers one hour bucket based on eventTime.',
		parts: {},
		totalRows: 0,
		updatedAt: new Date().toISOString(),
	}
}

export function resolveRowEventTime(row: Record<string, unknown>, fallback = new Date()): Date {
	const raw = row.eventTime ?? row.event_time ?? row.timestamp
	if (raw instanceof Date && !Number.isNaN(raw.getTime())) return raw
	if (typeof raw === 'string' || typeof raw === 'number') {
		const parsed = new Date(raw)
		if (!Number.isNaN(parsed.getTime())) return parsed
	}
	return fallback
}

export function hourKeyFromDate(date: Date): string {
	const year = date.getUTCFullYear()
	const month = String(date.getUTCMonth() + 1).padStart(2, '0')
	const day = String(date.getUTCDate()).padStart(2, '0')
	const hour = String(date.getUTCHours()).padStart(2, '0')
	return `${year}-${month}-${day}T${hour}`
}

export function hourlyPartPath(hourKey: string): string {
	return `${HOURLY_PARTITION_PREFIX}/${hourKey}.csv`
}

export function groupRowsByHour(rows: Record<string, unknown>[]): Map<string, Record<string, unknown>[]> {
	const groups = new Map<string, Record<string, unknown>[]>()
	for (const row of rows) {
		const hourKey = hourKeyFromDate(resolveRowEventTime(row))
		const bucket = groups.get(hourKey) ?? []
		bucket.push(row)
		groups.set(hourKey, bucket)
	}
	return groups
}

export function mergeRowsByEventId(
	existing: Record<string, unknown>[],
	incoming: Record<string, unknown>[],
): Record<string, unknown>[] {
	if (existing.length === 0) return incoming
	const merged = [...existing]
	const seen = new Set(
		existing.map(row => String(row.eventId ?? row.event_id ?? row.Event_ID ?? '')).filter(Boolean),
	)
	for (const row of incoming) {
		const eventId = String(row.eventId ?? row.event_id ?? row.Event_ID ?? '')
		if (eventId && seen.has(eventId)) continue
		if (eventId) seen.add(eventId)
		merged.push(row)
	}
	return merged
}

export function summarizeManifest(manifest: HourlyCsvManifest): HourlyCsvManifest {
	const parts = Object.values(manifest.parts)
	manifest.totalRows = parts.reduce((sum, part) => sum + part.rowCount, 0)
	manifest.updatedAt = new Date().toISOString()
	return manifest
}

export function updateManifestPart(
	manifest: HourlyCsvManifest,
	summary: HourlyPartUploadSummary,
): HourlyCsvManifest {
	manifest.parts[summary.hourKey] = {
		path: summary.path,
		rowCount: summary.totalRows,
		hourKey: summary.hourKey,
		updatedAt: new Date().toISOString(),
	}
	return summarizeManifest(manifest)
}

export function formatHourKeyLabel(hourKey: string): string {
	const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2})$/.exec(hourKey)
	if (!match) return hourKey
	const [, year, month, day, hour] = match
	return `${year}-${month}-${day} ${hour}:00–${hour}:59 UTC`
}

export function buildDatasetDescription(manifest: HourlyCsvManifest): string {
	const partCount = Object.keys(manifest.parts).length
	return [
		'Flowmatic smart-city dataset with hourly UTC CSV partitions.',
		`${partCount} hour file(s), ${manifest.totalRows.toLocaleString()} total row(s).`,
		`Files live under ${manifest.prefix} and are indexed in ${HOURLY_MANIFEST_PATH}.`,
	].join(' ')
}

export function buildHourlyReadme(input: {
	manifest: HourlyCsvManifest
	pipelineRunId: string
	fullRepoId: string
	previewRows: Record<string, unknown>[]
}): string {
	const { manifest, pipelineRunId, fullRepoId, previewRows } = input
	const headers = previewRows.length > 0 ? Object.keys(previewRows[0]) : []
	const timestamp = manifest.updatedAt
	const sortedParts = Object.values(manifest.parts).sort((a, b) => a.hourKey.localeCompare(b.hourKey))

	const columnInfo = headers.map(col => {
		const values = previewRows.map(row => row[col]).filter(value => value !== null && value !== undefined)
		const sample = values.slice(0, 3)
		let type = 'string'
		if (values.every(value => typeof value === 'boolean')) type = 'bool'
		else if (values.every(value => Number.isInteger(value))) type = 'int64'
		else if (values.every(value => typeof value === 'number')) type = 'float64'
		return { name: col, type, sampleValues: sample }
	})

	const partitionTable =
		sortedParts.length === 0
			? '_No hourly parts uploaded yet._'
			: [
					'| Hour (UTC) | File | Rows | Updated |',
					'| --- | --- | ---: | --- |',
					...sortedParts.map(part =>
						[
							formatHourKeyLabel(part.hourKey),
							`\`${part.path}\``,
							part.rowCount.toLocaleString(),
							part.updatedAt,
						].join(' | '),
					),
				].join('\n')

	return `---
dataset_info:
  features:
${columnInfo.length > 0 ? columnInfo.map(col => `    - name: ${col.name}\n      dtype: ${col.type}`).join('\n') : '    []'}
  splits:
    - name: train
      num_examples: ${manifest.totalRows}
license: cc-by-4.0
task_categories:
- tabular-classification
tags:
- flowmatic
- smart-city
- cleaned-data
- hourly-partition
---

# Flowmatic Smart City Dataset

${buildDatasetDescription(manifest)}

Pipeline run: \`${pipelineRunId}\`  
Updated: ${timestamp}  
Partition scheme: \`${manifest.partitionScheme}\`  
Manifest: \`${HOURLY_MANIFEST_PATH}\`

## Hourly CSV layout

Rows are grouped by \`eventTime\` into one CSV per UTC hour:

- Directory: \`${manifest.prefix}\`
- File pattern: \`YYYY-MM-DDTHH.csv\` (example: \`${HOURLY_PARTITION_PREFIX}/2026-05-21T14.csv\`)
- Append exports merge only within the touched hour file(s), not the whole dataset.

${partitionTable}

## Preview (latest exported rows)

${
	headers.length === 0
		? '_No preview rows available yet._'
		: [
				`| ${headers.join(' | ')} |`,
				`| ${headers.map(() => '---').join(' | ')} |`,
				...previewRows
					.slice(-5)
					.reverse()
					.map(row => `| ${headers.map(col => JSON.stringify(row[col] ?? '')).join(' | ')} |`),
			].join('\n')
}

## Usage

\`\`\`python
from datasets import load_dataset

# Loads all hourly CSV parts as one split
ds = load_dataset("${fullRepoId}", data_files="data/hourly/*.csv", split="train")
\`\`\`
`
}
