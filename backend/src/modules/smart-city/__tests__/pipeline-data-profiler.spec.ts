import { PipelineDataProfilerService } from '../pipeline-data-profiler.service'

describe('pipeline-data-profiler.service', () => {
	const profiler = new PipelineDataProfilerService()

	it('detects geospatial fields in flattened payloads', () => {
		const profile = profiler.profileRows([
			{ latitude: 51.16, longitude: 71.43, pm25: 42, sensorType: 'air' },
			{ latitude: 51.17, longitude: 71.44, pm25: 38, sensorType: 'air' },
		])
		expect(profile.hasGeospatial).toBe(true)
		expect(profile.latField).toBe('latitude')
		expect(profile.lngField).toBe('longitude')
		expect(profile.domains).toContain('geospatial')
	})

	it('detects numeric and categorical domains', () => {
		const profile = profiler.profileRows([
			{ category: 'traffic', speed: 42, timestamp: '2026-05-26T10:00:00Z' },
			{ category: 'traffic', speed: 55, timestamp: '2026-05-26T11:00:00Z' },
		])
		expect(profile.domains).toEqual(expect.arrayContaining(['timeseries', 'categorical', 'numeric']))
	})
})
