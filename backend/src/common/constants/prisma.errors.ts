// src/common/constants/prisma.errors.ts

/**
 * PrismaClientKnownRequestError codes
 * @see https://www.prisma.io/docs/orm/reference/error-reference#error-codes
 */
export enum PrismaErrorCode {
	// Common
	AuthenticationFailed = 'P1000',
	CannotReachDatabase = 'P1001',
	DatabaseDoesNotExist = 'P1002',
	HotspotCannotOpenPort = 'P1003',
	DatabaseAccessDenied = 'P1004',

	// Schema validation & migrations
	AlreadyConnected = 'P1010',
	SchemaPathNotFound = 'P1011',
	// …(add any other P10xx from your version)…

	// Constraint violations
	ValueTooLongForColumn = 'P2000',
	RecordNotFoundFilter = 'P2001',
	UniqueConstraintViolation = 'P2002',
	ForeignKeyConstraintViolation = 'P2003',
	ConstraintFailed = 'P2004',
	InvalidFieldValue = 'P2005',
	FieldValidationError = 'P2006',
	DataValidationError = 'P2007',
	TransactionFailed = 'P2008',
	RawQueryError = 'P2009',
	NullabilityViolation = 'P2010',
	MissingRequiredArgument = 'P2011',
	// …(other P20xx up through your version)…

	// Record-not-found helpers
	RecordNotFound = 'P2025',
}

/**
 * Helper map if you ever want to look up a human-readable title/description
 * by code. Extend as needed.
 */
export const PrismaErrorMeta: Record<PrismaErrorCode, { title: string; description: string }> = {
	[PrismaErrorCode.AuthenticationFailed]: {
		title: 'Authentication Failed',
		description: 'The provided database credentials are not valid. Please check your DATABASE_URL.',
	},
	[PrismaErrorCode.CannotReachDatabase]: {
		title: 'Database Unreachable',
		description:
			'Prisma cannot connect to your database. Ensure the database server is running and reachable.',
	},
	[PrismaErrorCode.DatabaseDoesNotExist]: {
		title: 'Database Does Not Exist',
		description: 'The specified database does not exist.',
	},
	[PrismaErrorCode.HotspotCannotOpenPort]: {
		title: 'Port Unavailable',
		description: 'Prisma could not bind to the required port.',
	},
	[PrismaErrorCode.DatabaseAccessDenied]: {
		title: 'Access Denied',
		description: 'Permission denied when attempting to access the database.',
	},

	[PrismaErrorCode.ValueTooLongForColumn]: {
		title: 'Value Too Long',
		description: 'A provided value exceeds the column length in the database.',
	},
	[PrismaErrorCode.RecordNotFoundFilter]: {
		title: 'Record Not Found (filter)',
		description: 'No record matched the filter criteria.',
	},
	[PrismaErrorCode.UniqueConstraintViolation]: {
		title: 'Unique Constraint Violation',
		description: 'Tried to insert or update a value that violates a unique constraint.',
	},
	[PrismaErrorCode.ForeignKeyConstraintViolation]: {
		title: 'Foreign Key Constraint Violation',
		description: 'Tried to insert or update a value that violates a foreign key constraint.',
	},
	[PrismaErrorCode.ConstraintFailed]: {
		title: 'Constraint Failed',
		description: 'A database constraint failed. See details in error metadata.',
	},
	[PrismaErrorCode.InvalidFieldValue]: {
		title: 'Invalid Field Value',
		description: 'A value stored in the database was invalid for its field type.',
	},
	[PrismaErrorCode.FieldValidationError]: {
		title: 'Field Validation Error',
		description: 'Prisma validation failed for one of the fields.',
	},
	[PrismaErrorCode.DataValidationError]: {
		title: 'Data Validation Error',
		description: 'General data validation error occurred.',
	},
	[PrismaErrorCode.TransactionFailed]: {
		title: 'Transaction Failed',
		description: 'A SQL transaction failed. See details in error metadata.',
	},
	[PrismaErrorCode.RawQueryError]: {
		title: 'Raw Query Error',
		description: 'An error occurred while running a raw SQL query.',
	},
	[PrismaErrorCode.NullabilityViolation]: {
		title: 'Nullability Violation',
		description: 'A non-nullable field was set to null.',
	},
	[PrismaErrorCode.MissingRequiredArgument]: {
		title: 'Missing Required Argument',
		description: 'A required argument was not provided.',
	},

	[PrismaErrorCode.RecordNotFound]: {
		title: 'Record Not Found',
		description:
			'The record you attempted to update or delete does not exist (use findUniqueOrThrow to auto-throw).',
	},
	[PrismaErrorCode.AlreadyConnected]: {
		title: '',
		description: '',
	},
	[PrismaErrorCode.SchemaPathNotFound]: {
		title: '',
		description: '',
	},
}
