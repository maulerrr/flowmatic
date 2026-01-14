import type { Request, Response, NextFunction } from 'express'

/**
 * Returns an express middleware that sets a longer socket timeout for
 * long-running generation endpoints (paths containing `/generate`).
 *
 * Usage: app.use(generationTimeoutMiddleware())
 */
export function generationTimeoutMiddleware() {
	try {
		const timeoutMs = Number(process.env.REQUEST_TIMEOUT_MS ?? '300000')
		if (Number.isNaN(timeoutMs) || timeoutMs <= 0) {
			// no-op middleware
			return (_req: Request, _res: Response, next: NextFunction) => next()
		}

		return (req: Request, res: Response, next: NextFunction) => {
			const path = req.path || req.url || ''
			if (/\/generate(\/|$)/.test(String(path))) {
				const maybeRes = res as unknown as { setTimeout?: (msec: number) => void }
				if (typeof maybeRes.setTimeout === 'function') {
					maybeRes.setTimeout(timeoutMs)
				}
				console.log(`Set per-request timeout ${timeoutMs}ms for ${path}`)
			}
			next()
		}
	} catch (err) {
		console.warn('Could not create generation timeout middleware', err)
		return (_req: Request, _res: Response, next: NextFunction) => next()
	}
}
