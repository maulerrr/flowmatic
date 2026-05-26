import { Injectable } from '@nestjs/common'
import { CopilotVizSpec } from './pipeline-copilot.types'
import {
	InsightVizId,
	PipelineAnalysisFindings,
	PipelineDataProfile,
} from './pipeline-insight.types'

type InsightBundle = {
	dataFlow: {
		hourlyEvents: Array<{ hour: string; count: number }>
		hourlyExportRows: Array<{ hour: string; rows: number }>
		stageCounts: { raw: number; cleaned: number; business: number }
		qualitySample?: { missingPct: number; duplicatePct: number; outlierPct: number }
	}
	exports: {
		successRuns24h: number
		failedRuns24h: number
		recentRuns: Array<{ adapterType: string; status: string; rowCount: number }>
	}
	sources: {
		items: Array<{ name: string; status: string }>
	}
	coreUnit: {
		activeModelLabel: string
		autoCleaning: boolean
		anomalyDetection: boolean
	}
	buildStandardViz: (chipId: string) => CopilotVizSpec
}

@Injectable()
export class PipelineInsightVizBuilderService {
	buildAll(
		visualizationIds: InsightVizId[],
		bundle: InsightBundle,
		profile: PipelineDataProfile,
		findings: PipelineAnalysisFindings,
		geoPoints: Array<{ lat: number; lng: number; weight?: number }>,
		scatterPair?: { xField: string; yField: string; points: Array<{ x: number; y: number }> },
	): CopilotVizSpec[] {
		const built = new Map<string, CopilotVizSpec>()
		for (const id of visualizationIds) {
			if (built.has(id)) continue
			built.set(id, this.buildOne(id, bundle, profile, findings, geoPoints, scatterPair))
		}
		return [...built.values()]
	}

	private buildOne(
		id: InsightVizId,
		bundle: InsightBundle,
		profile: PipelineDataProfile,
		findings: PipelineAnalysisFindings,
		geoPoints: Array<{ lat: number; lng: number; weight?: number }>,
		scatterPair?: { xField: string; yField: string; points: Array<{ x: number; y: number }> },
	): CopilotVizSpec {
		switch (id) {
			case 'event_volume':
				return bundle.buildStandardViz('event_volume')
			case 'export_health':
				return bundle.buildStandardViz('export_health')
			case 'data_funnel':
				return bundle.buildStandardViz('data_funnel')
			case 'source_health':
				return bundle.buildStandardViz('sources_status')
			case 'quality_kpis':
				return bundle.buildStandardViz('data_quality')
			case 'export_adapters':
				return bundle.buildStandardViz('export_adapters')
			case 'geo_map':
				return this.buildGeoMap(profile, geoPoints)
			case 'geo_heatmap':
				return this.buildGeoHeatmap(profile, findings)
			case 'metric_scatter':
				return this.buildScatter(scatterPair, findings)
			case 'multi_timeline':
				return this.buildTimeline(bundle)
			default:
				return bundle.buildStandardViz('event_volume')
		}
	}

	private buildGeoMap(
		profile: PipelineDataProfile,
		points: Array<{ lat: number; lng: number; weight?: number }>,
	): CopilotVizSpec {
		const bbox = profile.geoBounds ?? {
			minLat: 51.08,
			maxLat: 51.2,
			minLng: 71.34,
			maxLng: 71.57,
			pointCount: points.length,
		}
		const center = {
			lat: (bbox.minLat + bbox.maxLat) / 2,
			lng: (bbox.minLng + bbox.maxLng) / 2,
		}
		return {
			id: 'geo_map',
			title: 'Astana geospatial map',
			subtitle: `${points.length} sampled traffic points · semi-synthetic stream`,
			type: 'map',
			labels: [],
			series: [],
			meta: {
				points: points.slice(0, 250),
				bbox,
				center,
				city: 'Astana',
			},
		}
	}

	private buildGeoHeatmap(profile: PipelineDataProfile, findings: PipelineAnalysisFindings): CopilotVizSpec {
		const hotspots = findings.geoHotspots
		const bbox = profile.geoBounds ?? {
			minLat: 51.08,
			maxLat: 51.2,
			minLng: 71.34,
			maxLng: 71.57,
			pointCount: hotspots.length,
		}
		const maxWeight = Math.max(...hotspots.map(item => item.weight), 1)
		return {
			id: 'geo_heatmap',
			title: 'Astana spatial heatmap',
			subtitle: `${hotspots.length} activity clusters across the city grid`,
			type: 'map',
			labels: [],
			series: [],
			meta: {
				mode: 'heatmap',
				city: 'Astana',
				bbox,
				center: {
					lat: (bbox.minLat + bbox.maxLat) / 2,
					lng: (bbox.minLng + bbox.maxLng) / 2,
				},
				hotspots: hotspots.map(item => ({
					...item,
					intensity: Number(((item.weight / maxWeight) * 100).toFixed(1)),
				})),
			},
		}
	}

	private buildScatter(
		scatterPair:
			| { xField: string; yField: string; points: Array<{ x: number; y: number }> }
			| undefined,
		findings: PipelineAnalysisFindings,
	): CopilotVizSpec {
		const correlation = findings.correlations[0]
		const points = scatterPair?.points ?? []
		return {
			id: 'metric_scatter',
			title: 'Metric correlation scatter',
			subtitle: correlation
				? `${correlation.fieldA} vs ${correlation.fieldB} (r=${correlation.coefficient})`
				: 'Numeric field relationship',
			type: 'scatter',
			labels: [],
			series: [],
			meta: {
				xField: scatterPair?.xField ?? correlation?.fieldA ?? 'x',
				yField: scatterPair?.yField ?? correlation?.fieldB ?? 'y',
				points: points.slice(0, 200),
			},
		}
	}

	private buildTimeline(bundle: InsightBundle): CopilotVizSpec {
		const labels = bundle.dataFlow.hourlyEvents.map(bucket => bucket.hour)
		const exportByHour = new Map(bundle.dataFlow.hourlyExportRows.map(bucket => [bucket.hour, bucket.rows]))
		return {
			id: 'multi_timeline',
			title: 'Ingest vs export timeline',
			subtitle: 'Hourly events and exported rows (last 12h UTC)',
			type: 'timeline',
			labels,
			series: [
				{
					name: 'Events',
					values: bundle.dataFlow.hourlyEvents.map(bucket => bucket.count),
					color: '#34d0c3',
				},
				{
					name: 'Export rows',
					values: labels.map(label => exportByHour.get(label) ?? 0),
					color: '#7c8cff',
				},
			],
		}
	}
}
