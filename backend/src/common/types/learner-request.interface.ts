// src/common/types/learner-request.interface.ts

import { Request } from 'express'

export interface LearnerClaims {
	id: number
	nativeLanguage?: string
}

export interface LearnerRequest extends Request {
	learner: LearnerClaims
}
