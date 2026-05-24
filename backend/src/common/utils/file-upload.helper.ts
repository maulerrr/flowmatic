// src/common/helpers/file-upload.helper.ts

import { BadRequestException } from '@nestjs/common'

interface UploadedFileLike {
	mimetype: string
	originalname?: string
	filename?: string
}

interface UploadConfig {
	limits: {
		fileSize: number
	}
	fileFilter: (
		req: unknown,
		file: UploadedFileLike,
		cb: (error: Error | null, acceptFile: boolean) => void,
	) => void
}

/**
 * Image file filter for multer
 */
export const imageFileFilter = (
	req: unknown,
	file: UploadedFileLike,
	cb: (error: Error | null, acceptFile: boolean) => void,
) => {
	const allowedMimes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']

	if (allowedMimes.includes(file.mimetype)) {
		cb(null, true)
	} else {
		cb(
			new BadRequestException(
				'Invalid file type. Only JPEG, PNG, GIF, and WebP images are allowed.',
			),
			false,
		)
	}
}

/**
 * Standard image upload configuration for assessments
 */
export const createImageUploadConfig = (maxSizeMB: number = 5): UploadConfig => ({
	limits: {
		fileSize: maxSizeMB * 1024 * 1024, // Convert MB to bytes
	},
	fileFilter: imageFileFilter,
})

/**
 * Common file size limits
 */
export const FILE_SIZE_LIMITS = {
	SMALL: 1, // 1MB
	MEDIUM: 5, // 5MB
	LARGE: 10, // 10MB
} as const

/** ZIP file filter for multer (used for bulk imports). */
export const zipFileFilter = (
	req: unknown,
	file: UploadedFileLike,
	cb: (error: Error | null, acceptFile: boolean) => void,
) => {
	const name = file.originalname ?? file.filename ?? ''
	const nameOk = name.toLowerCase().endsWith('.zip')
	const mimeOk = file.mimetype.toLowerCase().includes('zip')

	if (nameOk || mimeOk) {
		cb(null, true)
		return
	}

	cb(new BadRequestException('Invalid file type. Only .zip files are allowed.'), false)
}

/** Standard zip upload configuration (in-memory). */
export const createZipUploadConfig = (maxSizeMB: number = 50): UploadConfig => ({
	limits: {
		fileSize: maxSizeMB * 1024 * 1024,
	},
	fileFilter: zipFileFilter,
})
