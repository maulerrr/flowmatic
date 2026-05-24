// src/common/types/learner-request.interface.ts

import { FastifyRequest } from 'fastify'

export interface LearnerClaims {
	id: number
	nativeLanguage?: string
}

export interface LearnerRequest extends FastifyRequest {
	learner: LearnerClaims
}
