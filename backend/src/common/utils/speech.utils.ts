import { createHash } from 'crypto'

/** Normalize text for cache keys and TTS parity */
export function normalizeTtsText(input: string): string {
	if (!input) return ''
	const nk = input.normalize('NFKC')
	const trimmed = nk.trim()
	const collapsed = trimmed.replace(/\s+/g, ' ')
	return collapsed
}

/** Build deterministic S3 key path for TTS audio (relative to audioPrefix). */
export function buildTtsS3Key(
	voice: string,
	version: string,
	normalizedText: string,
	ext: string,
): string {
	const hash = createHash('sha256').update(`${voice}|${version}|${normalizedText}`).digest('hex')
	const shard = `${hash.substring(0, 2)}/${hash.substring(2, 4)}`
	const safeVoice = voice.replace(/[^a-zA-Z0-9_-]/g, '_') || 'default'
	const v = version.replace(/[^a-zA-Z0-9_-]/g, '_') || 'v1'
	return `tts/${safeVoice}/${v}/${shard}/${hash}.${ext}`
}
