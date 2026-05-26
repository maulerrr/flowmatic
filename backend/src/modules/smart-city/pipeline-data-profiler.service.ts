import { Injectable } from '@nestjs/common'
import { DataRow } from 'src/common/types/data.types'
import { DataFieldKind, DataFieldProfile, PipelineDataProfile } from './pipeline-insight.types'

const LAT_KEYS = ['latitude', 'lat', 'y']
const LNG_KEYS = ['longitude', 'lng', 'lon', 'long', 'x']
const TIME_KEYS = ['timestamp', 'eventtime', 'event_time', 'time', 'datetime', 'createdat', 'created_at']

function normalizeKey(key: string) {
	return key.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function flattenRecord(input: Record<string, unknown>, prefix = ''): Record<string, unknown> {
	const flat: Record<string, unknown> = {}
	for (const [key, value] of Object.entries(input)) {
		const path = prefix ? `${prefix}.${key}` : key
		if (value && typeof value === 'object' && !Array.isArray(value)) {
			Object.assign(flat, flattenRecord(value as Record<string, unknown>, path))
		} else {
			flat[path] = value
		}
	}
	return flat
}

function inferKind(name: string, values: unknown[]): DataFieldKind {
	const key = normalizeKey(name.split('.').pop() ?? name)
	if (LAT_KEYS.includes(key)) return 'latitude'
	if (LNG_KEYS.includes(key)) return 'longitude'
	if (TIME_KEYS.includes(key) || key.endsWith('at')) return 'timestamp'

	const numericValues = values.filter(value => typeof value === 'number') as number[]
	if (numericValues.length >= Math.max(1, values.length * 0.6)) {
		if (numericValues.every(value => Number.isInteger(value) && Math.abs(value) > 1000)) return 'id'
		return 'numeric'
	}

	const stringValues = values.filter(value => typeof value === 'string') as string[]
	if (stringValues.length >= Math.max(1, values.length * 0.6)) {
		const uniqueRatio = new Set(stringValues).size / Math.max(stringValues.length, 1)
		if (uniqueRatio <= 0.35 || stringValues.length <= 12) return 'category'
		return 'text'
	}

	return 'unknown'
}

@Injectable()
export class PipelineDataProfilerService {
	profileRows(
		rows: Array<Record<string, unknown>>,
		meta?: { sensorTypes?: string[]; sourceCount?: number },
	): PipelineDataProfile {
		const flattenedRows = rows.map(row => flattenRecord(row))
		const columnValues = new Map<string, unknown[]>()

		for (const row of flattenedRows) {
			for (const [key, value] of Object.entries(row)) {
				if (!columnValues.has(key)) columnValues.set(key, [])
				columnValues.get(key)!.push(value ?? null)
			}
		}

		const fields: DataFieldProfile[] = [...columnValues.entries()]
			.map(([name, values]) => {
				const nonNull = values.filter(value => value !== null && value !== undefined && value !== '')
				const numeric = nonNull.filter(value => typeof value === 'number') as number[]
				const kind = inferKind(name, nonNull)
				return {
					name,
					kind,
					sampleValues: nonNull.slice(0, 5).map(value =>
						typeof value === 'object' ? JSON.stringify(value) : (value as string | number | null),
					),
					nonNullPct: rows.length ? Number(((nonNull.length / rows.length) * 100).toFixed(1)) : 0,
					uniqueCount: new Set(nonNull.map(value => String(value))).size,
					min: numeric.length ? Math.min(...numeric) : undefined,
					max: numeric.length ? Math.max(...numeric) : undefined,
				}
			})
			.sort((a, b) => b.nonNullPct - a.nonNullPct)
			.slice(0, 40)

		const latField = fields.find(field => field.kind === 'latitude')?.name
		const lngField = fields.find(field => field.kind === 'longitude')?.name
		const timeField = fields.find(field => field.kind === 'timestamp')?.name
		const hasGeospatial = Boolean(latField && lngField)
		const hasTimeseries = Boolean(timeField) || fields.some(field => field.kind === 'timestamp')

		const domains: PipelineDataProfile['domains'] = []
		if (hasGeospatial) domains.push('geospatial')
		if (hasTimeseries) domains.push('timeseries')
		if (fields.some(field => field.kind === 'category')) domains.push('categorical')
		if (fields.some(field => field.kind === 'numeric')) domains.push('numeric')
		if (fields.some(field => field.kind === 'text')) domains.push('text')

		let geoBounds: PipelineDataProfile['geoBounds']
		if (hasGeospatial && latField && lngField) {
			const points = flattenedRows
				.map(row => ({
					lat: Number(row[latField]),
					lng: Number(row[lngField]),
				}))
				.filter(point => Number.isFinite(point.lat) && Number.isFinite(point.lng))
			if (points.length) {
				geoBounds = {
					minLat: Math.min(...points.map(point => point.lat)),
					maxLat: Math.max(...points.map(point => point.lat)),
					minLng: Math.min(...points.map(point => point.lng)),
					maxLng: Math.max(...points.map(point => point.lng)),
					pointCount: points.length,
				}
			}
		}

		return {
			generatedAt: new Date().toISOString(),
			sampleSize: rows.length,
			rowCount: rows.length,
			fields,
			domains,
			hasGeospatial,
			hasTimeseries,
			geoBounds,
			latField,
			lngField,
			timeField,
			sensorTypes: meta?.sensorTypes ?? [],
			sourceCount: meta?.sourceCount ?? 0,
		}
	}

	profileMixedSamples(input: {
		eventPayloads: Array<Record<string, unknown>>
		cleanedRows?: DataRow[]
		sensorTypes?: string[]
		sourceCount?: number
	}): PipelineDataProfile {
		const cleanedAsRecords = (input.cleanedRows ?? []).map(row => ({ ...row }))
		const merged = [...input.eventPayloads, ...cleanedAsRecords].slice(0, 500)
		return this.profileRows(merged, {
			sensorTypes: input.sensorTypes,
			sourceCount: input.sourceCount,
		})
	}
}
