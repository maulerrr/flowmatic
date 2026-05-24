export const SENSOR_KINDS = ['iot', 'video', 'power', 'network', 'weather', 'parking'] as const
export type SensorKind = (typeof SENSOR_KINDS)[number]

export interface SensorPayload {
	location: string
	sensorKind: SensorKind
	transport: 'WEBSOCKET' | 'HTTP_POLLING'
	timestamp: string
	confidence: number
	[key: string]: unknown
}

export function isSensorKind(value: string | null | undefined): value is SensorKind {
	return SENSOR_KINDS.includes(value as SensorKind)
}

export function createPayload(
	sensorKind: SensorKind,
	transport: SensorPayload['transport'],
	location: string,
): SensorPayload {
	const base = {
		location,
		sensorKind,
		transport,
		timestamp: new Date().toISOString(),
		confidence: Number((0.82 + Math.random() * 0.17).toFixed(3)),
	}

	switch (sensorKind) {
		case 'video':
			return {
				...base,
				vehicleCount: Math.floor(20 + Math.random() * 180),
				averageSpeedKph: Math.floor(20 + Math.random() * 55),
			}
		case 'power':
			return {
				...base,
				loadPercent: Math.floor(35 + Math.random() * 60),
				voltage: Number((218 + Math.random() * 18).toFixed(1)),
			}
		case 'network':
			return {
				...base,
				latencyMs: Math.floor(8 + Math.random() * 60),
				packetLossPercent: Number((Math.random() * 1.8).toFixed(2)),
			}
		case 'weather':
			return {
				...base,
				temperatureC: Number((-8 + Math.random() * 35).toFixed(1)),
				windKph: Number((2 + Math.random() * 30).toFixed(1)),
			}
		case 'parking':
			return {
				...base,
				occupancyPercent: Math.floor(20 + Math.random() * 78),
				openSpaces: Math.floor(Math.random() * 120),
			}
		default:
			return {
				...base,
				airQualityIndex: Math.floor(20 + Math.random() * 130),
				pm25: Number((3 + Math.random() * 45).toFixed(1)),
			}
	}
}

export function createBatch(
	sensorKind: SensorKind,
	transport: SensorPayload['transport'],
	location: string,
	limit: number,
) {
	const count = Math.min(Math.max(limit, 1), 50)
	return Array.from({ length: count }, () => createPayload(sensorKind, transport, location))
}
