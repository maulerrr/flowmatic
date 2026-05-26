import { Injectable, Logger, OnModuleInit } from '@nestjs/common'
import { Interval } from '@nestjs/schedule'
import { BossService } from 'src/common/queue/boss.service'
import { AuthContext } from 'src/modules/auth/auth-context.service'
import { PipelineInsightService } from './pipeline-insight.service'

export const SMART_CITY_INSIGHT_QUEUE = 'smart-city-pipeline-insight'

interface InsightJobPayload {
	pipelineId: string
	organizationId: string
	userId: string
	trigger: 'scheduled' | 'manual'
}

@Injectable()
export class PipelineInsightScheduler implements OnModuleInit {
	private readonly logger = new Logger(PipelineInsightScheduler.name)
	private ticking = false

	constructor(
		private readonly boss: BossService,
		private readonly insights: PipelineInsightService,
	) {}

	async onModuleInit() {
		try {
			await this.boss.subscribe(SMART_CITY_INSIGHT_QUEUE, async job => {
				const payload = (job as { data?: InsightJobPayload })?.data
				if (!payload?.pipelineId || !payload.organizationId || !payload.userId) return
				const scope: AuthContext = {
					userId: payload.userId,
					organizationId: payload.organizationId,
					role: 'admin',
				}
				await this.insights.executeRun(scope, payload.pipelineId, payload.trigger)
			})
			this.logger.log('Pipeline insight queue subscriber initialized')
		} catch (error) {
			this.logger.warn('Pipeline insight queue unavailable', error)
		}
	}

	@Interval(60_000)
	async enqueueDueInsightRuns() {
		if (this.ticking) return
		this.ticking = true
		try {
			const due = await this.insights.listDuePipelines()
			for (const pipeline of due) {
				const running = await this.insights.hasRunningInsightRun(pipeline.id)
				if (running) continue
				await this.boss.publish(SMART_CITY_INSIGHT_QUEUE, {
					pipelineId: pipeline.id,
					organizationId: pipeline.organizationId,
					userId: pipeline.createdByUserId,
					trigger: 'scheduled',
				} satisfies InsightJobPayload)
			}
			if (due.length) {
				this.logger.log(`Enqueued ${due.length} scheduled pipeline insight run(s)`)
			}
		} catch (error) {
			this.logger.warn('Failed to enqueue scheduled insight runs', error)
		} finally {
			this.ticking = false
		}
	}
}
