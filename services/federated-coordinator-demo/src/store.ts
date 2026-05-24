interface FederatedMessage {
	type?: string
	projectId?: string | null
	nodeId?: string | null
	topic?: string | null
	registrationId?: string | null
	currentRoundId?: string | null
	globalModelVersion?: string | null
	timestamp?: string
	data?: Record<string, unknown>
}

export class CoordinatorStore {
	globalModelVersion = 'global-v1.0.0'
	currentRoundId: string | null = null
	registrationCount = 0
	private registrations = new Map<string, string>()
	private rounds: Array<Record<string, unknown>> = []
	private receivedEvents: Array<Record<string, unknown>> = []

	register(nodeId: string) {
		this.registrationCount += 1
		const registrationId = `reg-${nodeId}-${this.registrationCount}`
		this.registrations.set(nodeId, registrationId)
		return {
			registrationId,
			globalModelVersion: this.globalModelVersion,
			currentRoundId: this.currentRoundId,
			summary: `Registered node ${nodeId}`,
		}
	}

	handleMessage(message: FederatedMessage) {
		const type = String(message.type ?? 'unknown')
		const nodeId = message.nodeId ?? 'unknown-node'
		this.receivedEvents.unshift({ type, nodeId, at: new Date().toISOString(), data: message.data ?? {} })
		this.receivedEvents = this.receivedEvents.slice(0, 100)

		switch (type) {
			case 'register':
				return this.register(nodeId)
			case 'pull_global_model':
				return {
					globalModelVersion: this.globalModelVersion,
					currentRoundId: this.currentRoundId,
					rounds: this.rounds.slice(0, 10),
					summary: 'Global model state returned',
				}
			case 'round_started': {
				const roundId = String(message.data?.roundId ?? `round-${Date.now()}`)
				this.currentRoundId = roundId
				const round = {
					id: roundId,
					name: message.data?.name ?? 'Demo round',
					status: 'ACTIVE',
					startedAt: new Date().toISOString(),
					sampleCount: message.data?.sampleCount ?? null,
					participants: [],
				}
				this.rounds.unshift(round)
				this.rounds = this.rounds.slice(0, 20)
				return { ok: true, roundId, summary: 'Round started on demo coordinator' }
			}
			case 'round_update_submitted':
				return {
					ok: true,
					roundId: message.data?.roundId ?? this.currentRoundId,
					summary: 'Update recorded',
				}
			case 'round_aggregated': {
				const version = String(message.data?.globalModelVersion ?? `global-${Date.now()}`)
				this.globalModelVersion = version
				this.currentRoundId = null
				return { ok: true, globalModelVersion: version, summary: 'Round aggregated' }
			}
			default:
				return {
					ok: true,
					type,
					summary: `Acknowledged ${type}`,
					globalModelVersion: this.globalModelVersion,
					currentRoundId: this.currentRoundId,
				}
		}
	}
}
