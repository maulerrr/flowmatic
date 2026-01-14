// src/common/pipes/parse-json.pipe.ts
import { PipeTransform, Injectable, BadRequestException } from '@nestjs/common'

/**
 * Parses a JSON-encoded string into an object or array.
 * Returns the original value if it isn’t a string.
 * Always returns `unknown`, never `any`.
 */
@Injectable()
export class ParseJsonPipe implements PipeTransform<string, unknown> {
	transform(value: string): unknown {
		if (typeof value !== 'string') {
			// non-strings just pass through
			return value
		}

		let parsed: unknown
		try {
			parsed = JSON.parse(value)
		} catch {
			throw new BadRequestException('Invalid JSON')
		}
		return parsed
	}
}
