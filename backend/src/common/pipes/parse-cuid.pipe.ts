import { BadRequestException, Injectable, PipeTransform } from '@nestjs/common'

const CUID_REGEX = /^c[a-z0-9]{20,30}$/

@Injectable()
export class ParseCuidPipe implements PipeTransform<string, string> {
	transform(value: string): string {
		if (!value || !CUID_REGEX.test(value)) {
			throw new BadRequestException('Invalid identifier')
		}

		return value
	}
}
