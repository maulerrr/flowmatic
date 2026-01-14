import { addSeconds, addMinutes, addHours, addDays } from 'date-fns'

/**
 * Turn an expiresIn string like "24h", "45m", "7d" or "30s"
 * into a Date relative to `base` (defaults to now).
 *
 * @param expiresIn  number + unit, where unit is one of:
 *                   s (seconds), m (minutes), h (hours), d (days)
 * @param base       a Date to offset from (default: new Date())
 */
export function parseExpirationDate(expiresIn: string, base: Date = new Date()): Date {
	const match = expiresIn.match(/^(\d+)([smhd])$/)
	if (!match) {
		throw new Error(`Invalid expiresIn format: "${expiresIn}"`)
	}

	const value = parseInt(match[1], 10)
	const unit = match[2]

	switch (unit) {
		case 's':
			return addSeconds(base, value)
		case 'm':
			return addMinutes(base, value)
		case 'h':
			return addHours(base, value)
		case 'd':
			return addDays(base, value)
		default:
			// TS should never get here because of the regex
			throw new Error(`Unsupported time unit "${unit}" in expiresIn`)
	}
}
