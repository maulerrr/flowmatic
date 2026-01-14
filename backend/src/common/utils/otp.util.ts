import bcrypt from 'bcrypt'

const SALT_ROUNDS = 10

/**
 * Hashes a plain OTP using bcrypt.
 */
export function hashOtp(otp: string): Promise<string> {
	return bcrypt.hash(otp, SALT_ROUNDS)
}

/**
 * Compares a plain OTP against a bcrypt hash.
 */
export function compareOtp(otp: string, hash: string): Promise<boolean> {
	return bcrypt.compare(otp, hash)
}

/**
 * Generates a random 4-digit OTP code as a string.
 * You can specify the length (default is 4).
 */
export function generateOtp(n: number = 4): string {
	const min = Math.pow(10, n - 1)
	const max = Math.pow(10, n) - 1
	return Math.floor(Math.random() * (max - min + 1) + min).toString()
}
