import { existsSync, readFileSync } from 'fs'

export interface AstanaTrafficRow {
	eventId: string
	sourceTimestamp: string
	vehicleType: string
	speedKmh: number
	latitude: number
	longitude: number
	eventType: string
	severity: string
	trafficDensity: number
}

let rows: AstanaTrafficRow[] = []
let cursor = 0

export function loadAstanaDataset(datasetPath: string) {
	rows = []
	cursor = 0
	if (!existsSync(datasetPath)) {
		console.warn(`[astana-traffic] Dataset not found at ${datasetPath}`)
		return 0
	}

	const text = readFileSync(datasetPath, 'utf8')
	const lines = text.trim().split(/\r?\n/)
	for (const line of lines.slice(1)) {
		const parts = line.split(',')
		if (parts.length < 9) continue
		const latitude = Number(parts[4])
		const longitude = Number(parts[5])
		if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue
		rows.push({
			eventId: parts[0],
			sourceTimestamp: parts[1],
			vehicleType: parts[2],
			speedKmh: Number(parts[3]) || 0,
			latitude,
			longitude,
			eventType: parts[6],
			severity: parts[7],
			trafficDensity: Number(parts[8]) || 0,
		})
	}

	console.log(`[astana-traffic] Loaded ${rows.length} semi-synthetic source rows`)
	return rows.length
}

function nextRow(): AstanaTrafficRow {
	if (!rows.length) {
		return {
			eventId: '0',
			sourceTimestamp: new Date().toISOString(),
			vehicleType: 'Car',
			speedKmh: 45,
			latitude: 51.16,
			longitude: 71.43,
			eventType: 'Normal',
			severity: 'Low',
			trafficDensity: 60,
		}
	}
	const row = rows[cursor % rows.length]
	cursor += 1
	return row
}

function jitter(value: number, spread: number) {
	return Number((value + (Math.random() - 0.5) * spread).toFixed(6))
}

export function createAstanaTrafficPayload(
	transport: 'WEBSOCKET' | 'HTTP_POLLING',
	location: string,
) {
	const row = nextRow()
	const latitude = jitter(row.latitude, 0.004)
	const longitude = jitter(row.longitude, 0.004)
	const speedKmh = Math.max(0, Math.round(jitter(row.speedKmh, 6)))
	const trafficDensity = Number(jitter(row.trafficDensity, 4).toFixed(2))

	return {
		location,
		city: 'Astana',
		sensorKind: 'traffic' as const,
		transport,
		timestamp: new Date().toISOString(),
		sourceTimestamp: row.sourceTimestamp,
		confidence: Number((0.88 + Math.random() * 0.11).toFixed(3)),
		eventId: row.eventId,
		vehicleType: row.vehicleType,
		vehicle_type: row.vehicleType,
		speedKmh,
		speed_kmh: speedKmh,
		Speed_kmh: speedKmh,
		latitude,
		longitude,
		Latitude: latitude,
		Longitude: longitude,
		eventType: row.eventType,
		event_type: row.eventType,
		Event_Type: row.eventType,
		severity: row.severity,
		Severity: row.severity,
		trafficDensity,
		traffic_density: trafficDensity,
		Traffic_Density: trafficDensity,
		derivedFrom: 'astana_synthetic_data.csv',
		streamKind: 'astana_semi_synthetic',
	}
}

export function getAstanaDatasetStats() {
	if (!rows.length) return null
	const latitudes = rows.map(row => row.latitude)
	const longitudes = rows.map(row => row.longitude)
	return {
		rowCount: rows.length,
		bbox: {
			minLat: Math.min(...latitudes),
			maxLat: Math.max(...latitudes),
			minLng: Math.min(...longitudes),
			maxLng: Math.max(...longitudes),
		},
	}
}
