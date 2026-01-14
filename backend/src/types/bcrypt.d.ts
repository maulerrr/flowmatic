declare module 'bcrypt' {
	/**
	 * The same signature as bcrypt.hash
	 * @param data – the string to hash
	 * @param saltRounds – number of salt rounds
	 * @returns Promise<string>
	 */
	export function hash(data: string, saltRounds: number): Promise<string>

	/**
	 * The same signature as bcrypt.compare
	 * @param data – the plain text to compare
	 * @param encrypted – the existing hash
	 * @returns Promise<boolean>
	 */
	export function compare(data: string, encrypted: string): Promise<boolean>
}
