// src/common/utils/validation.util.ts
import 'reflect-metadata'
import { plainToInstance } from 'class-transformer'
import { validateSync, ValidationError } from 'class-validator'
import { BadRequestException } from '@nestjs/common'

/**
 * Transforms a plain object into a class instance and validates it.
 * Throws BadRequestException if validation fails.
 *
 * @param cls    The DTO class to instantiate
 * @param obj    The plain payload (e.g. req.body or parsed form-data)
 */
export function validateDto<T extends object>(cls: new () => T, obj: object): T {
	// 1) transform + apply @Type() / @Transform() decorators
	const dto = plainToInstance(cls, obj)

	// 2) run all sync validations
	const errors: ValidationError[] = validateSync(dto, {
		whitelist: true,
		forbidNonWhitelisted: true,
		skipMissingProperties: false,
	})

	if (errors.length) {
		throw new BadRequestException(errors)
	}

	return dto
}
