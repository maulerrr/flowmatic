import { Injectable } from '@nestjs/common'
import { InsightFinding, PipelineAnalysisFindings } from './pipeline-insight.types'

interface AnalysisInput {
	profile: import('./pipeline-insight.types').PipelineDataProfile
	eventsLast24h: number
	exportSuccess24h: number
	exportFailed24h: number
	sourceErrors: number
	runningSources: number
	totalSources: number
	hourlyEvents: Array<{ hour: string; count: number }>
	qualitySample?: { missingPct: number; duplicatePct: number; outlierPct: number }
	geoPoints: Array<{ lat: number; lng: number; weight?: number }>
	numericPairs: Array<{ xField: string; yField: string; points: Array<{ x: number; y: number }> }>
}

function pearson(points: Array<{ x: number; y: number }>): number {
	if (points.length < 3) return 0
	const xs = points.map(point => point.x)
	const ys = points.map(point => point.y)
	const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length
	const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length
	let num = 0
	let denX = 0
	let denY = 0
	for (let index = 0; index < points.length; index += 1) {
		const dx = xs[index] - meanX
		const dy = ys[index] - meanY
		num += dx * dy
		denX += dx * dx
		denY += dy * dy
	}
	if (!denX || !denY) return 0
	return Number((num / Math.sqrt(denX * denY)).toFixed(3))
}

function buildGeoHotspots(points: Array<{ lat: number; lng: number; weight?: number }>) {
	const buckets = new Map<string, { lat: number; lng: number; weight: number; count: number }>()
	for (const point of points) {
		const lat = Math.round(point.lat * 100) / 100
		const lng = Math.round(point.lng * 100) / 100
		const key = `${lat}:${lng}`
		const existing = buckets.get(key)
		if (existing) {
			existing.weight += point.weight ?? 1
			existing.count += 1
		} else {
			buckets.set(key, { lat, lng, weight: point.weight ?? 1, count: 1 })
		}
	}
	return [...buckets.values()]
		.sort((a, b) => b.weight - a.weight)
		.slice(0, 12)
		.map((bucket, index) => ({
			lat: bucket.lat,
			lng: bucket.lng,
			weight: bucket.weight,
			label: `Hotspot ${index + 1} (${bucket.count} pts)`,
		}))
}

@Injectable()
export class PipelineAnalysisEngineService {
	analyze(input: AnalysisInput): PipelineAnalysisFindings {
		const findings: InsightFinding[] = []
		const correlations: PipelineAnalysisFindings['correlations'] = []

		if (input.profile.hasGeospatial) {
			findings.push({
				id: 'geo-detected',
				category: 'geo',
				severity: 'info',
				title: 'Geospatial data detected',
				summary: `Found latitude/longitude fields (${input.profile.latField}, ${input.profile.lngField}) across ${input.profile.geoBounds?.pointCount ?? 0} sampled points — map and heatmap views are recommended.`,
				recommendedViz: 'geo_map',
				evidence: input.profile.geoBounds,
			})
		}

		if (input.profile.hasTimeseries) {
			const peak = [...input.hourlyEvents].sort((a, b) => b.count - a.count)[0]
			if (peak && peak.count > 0) {
				findings.push({
					id: 'hourly-peak',
					category: 'pattern',
					severity: 'info',
					title: 'Peak ingest hour identified',
					summary: `Highest event volume in the last 24h occurred around ${peak.hour} UTC with ${peak.count} events.`,
					recommendedViz: 'event_volume',
					evidence: { hour: peak.hour, count: peak.count },
				})
			}
		}

		if (input.qualitySample) {
			const { missingPct, duplicatePct, outlierPct } = input.qualitySample
			if (missingPct > 5 || duplicatePct > 3 || outlierPct > 5) {
				findings.push({
					id: 'quality-warning',
					category: 'quality',
					severity: missingPct > 10 || outlierPct > 10 ? 'critical' : 'warning',
					title: 'Data quality drift in cleaned sample',
					summary: `Cleaned sample shows ${missingPct}% missing, ${duplicatePct}% duplicates, ${outlierPct}% outliers.`,
					recommendedViz: 'quality_kpis',
					evidence: input.qualitySample,
				})
			}
		}

		if (input.exportFailed24h > 0) {
			findings.push({
				id: 'export-failures',
				category: 'ops',
				severity: input.exportFailed24h >= 3 ? 'critical' : 'warning',
				title: 'Export failures detected',
				summary: `${input.exportFailed24h} export runs failed in the last 24h versus ${input.exportSuccess24h} successes.`,
				recommendedViz: 'export_health',
			})
		}

		if (input.sourceErrors > 0) {
			findings.push({
				id: 'source-errors',
				category: 'ops',
				severity: 'warning',
				title: 'Ingest sources reporting errors',
				summary: `${input.sourceErrors} of ${input.totalSources} sources are in error while ${input.runningSources} are running.`,
				recommendedViz: 'source_health',
			})
		}

		for (const pair of input.numericPairs) {
			const coefficient = pearson(pair.points)
			if (Math.abs(coefficient) >= 0.65) {
				correlations.push({
					fieldA: pair.xField,
					fieldB: pair.yField,
					coefficient,
				})
				findings.push({
					id: `corr-${pair.xField}-${pair.yField}`,
					category: 'correlation',
					severity: Math.abs(coefficient) >= 0.85 ? 'warning' : 'info',
					title: `Correlation between ${pair.xField} and ${pair.yField}`,
					summary: `Pearson r=${coefficient} across ${pair.points.length} aligned samples.`,
					recommendedViz: 'metric_scatter',
					evidence: { fieldA: pair.xField, fieldB: pair.yField, coefficient },
				})
			}
		}

		const geoHotspots = buildGeoHotspots(input.geoPoints)
		if (geoHotspots.length >= 2) {
			findings.push({
				id: 'geo-hotspots',
				category: 'geo',
				severity: 'info',
				title: 'Spatial activity clusters',
				summary: `Identified ${geoHotspots.length} geospatial clusters in the sampled data.`,
				recommendedViz: 'geo_heatmap',
				evidence: { hotspots: geoHotspots.slice(0, 5) },
			})
		}

		const featureEngineering: PipelineAnalysisFindings['featureEngineering'] = []
		if (input.profile.hasGeospatial && input.profile.timeField) {
			featureEngineering.push({
				name: 'geo_hour_density',
				formula: 'count(events) GROUP BY round(lat,2), round(lng,2), hour(timestamp)',
				purpose: 'Surface spatial-temporal hotspots for smart-city routing and anomaly detection.',
			})
		}
		if (input.profile.fields.some(field => field.kind === 'numeric')) {
			featureEngineering.push({
				name: 'rolling_zscore',
				formula: 'zscore(metric, window=24h)',
				purpose: 'Detect metric spikes relative to recent baseline.',
			})
		}

		return {
			generatedAt: new Date().toISOString(),
			findings,
			correlations,
			geoHotspots,
			hourlyPattern: input.hourlyEvents,
			featureEngineering,
		}
	}
}
