import { PipelineModelRegistryService } from '../pipeline-model-registry.service'
import { PipelineModelRouterService } from '../pipeline-model-router.service'

describe('PipelineModelRouterService', () => {
	const registry = {
		listModels: jest.fn(),
		findById: jest.fn(),
	} as unknown as PipelineModelRegistryService
	const router = new PipelineModelRouterService(registry)

	beforeEach(() => {
		jest.clearAllMocks()
		;(registry.listModels as jest.Mock).mockReturnValue([
			{
				id: 'research:q1_astana_tranad_anomaly_detector_seed42',
				label: 'tranad',
				kind: 'tranad_anomaly',
				dataset: 'astana',
				modality: 'traffic',
				tasks: ['anomaly'],
				sensorKinds: ['traffic', 'generic'],
				priority: 1,
				source: 'manifest',
			},
			{
				id: 'research:q1_hf_weather_timesblock_forecaster_seed2026',
				label: 'weather',
				kind: 'timesblock_forecast',
				dataset: 'hf_weather',
				modality: 'weather',
				tasks: ['forecast'],
				sensorKinds: ['weather', 'air'],
				priority: 1,
				source: 'manifest',
			},
			{
				id: 'research:q1_astana_saits_imputer_seed42',
				label: 'saits',
				kind: 'saits_imputer',
				dataset: 'astana',
				modality: 'generic',
				tasks: ['imputation'],
				sensorKinds: ['traffic', 'weather', 'air', 'generic'],
				priority: 8,
				source: 'manifest',
			},
		])
	})

	it('routes traffic payloads away from weather models in auto mode', () => {
		const decision = router.resolveModel({
			mode: 'auto',
			manualModelId: null,
			sensorKind: 'traffic',
			payload: { speedKmh: 42, latitude: 51.1, longitude: 71.4, trafficDensity: 70 },
			anomalyDetection: true,
		})

		expect(decision.modelId).toContain('tranad')
		expect(decision.profile.modality).toBe('traffic')
	})

	it('routes weather payloads to weather models in auto mode', () => {
		const decision = router.resolveModel({
			mode: 'auto',
			manualModelId: null,
			sensorKind: 'weather',
			payload: { temperatureC: 3.2, humidityPct: 81, pressureHpa: 1010 },
			anomalyDetection: true,
		})

		expect(decision.modelId).toContain('weather')
		expect(decision.profile.modality).toBe('weather')
	})

	it('routes sparse traffic payloads to generic imputation models', () => {
		const decision = router.resolveModel({
			mode: 'auto',
			manualModelId: null,
			sensorKind: 'traffic',
			payload: { speedKmh: null, trafficDensity: null, latitude: 51.1, longitude: 71.4 },
			anomalyDetection: false,
		})

		expect(decision.modelId).toContain('saits')
		expect(decision.profile.preferredTask).toBe('imputation')
	})

	it('uses manual model id in manual mode', () => {
		const decision = router.resolveModel({
			mode: 'manual',
			manualModelId: 'hf:pushthetempo/custom-model',
			sensorKind: 'traffic',
			payload: { speedKmh: 40 },
		})

		expect(decision.modelId).toBe('hf:pushthetempo/custom-model')
		expect(decision.plannerSource).toBe('manual')
	})
})
