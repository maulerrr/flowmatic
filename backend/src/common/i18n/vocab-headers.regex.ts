const SEP = String.raw`[\s_\-./(){}\[\]—–:·|]*`
const RE = (s: string) => new RegExp(s, 'iu')

/** Order matters: most specific first, all ^...$ anchored */
const HEADER_PATTERNS = {
	word: [RE(`^(?:сөз|слово|word)$`)],
	example: [RE(`^(?:мысал|пример|example)$`)],
	translation_kz: [
		RE(`^(?:қазақша|казахский|kazakh|kz)$`),
		RE(`^(?:перевод|translation)${SEP}(?:kz|kazakh|на${SEP}казахский)$`),
		RE(`^(?:қазақша|казахский|kazakh|kz)${SEP}(?:аударма|перевод|translation)$`),
	],
	translation_ru: [
		RE(`^(?:орысша|русский|russian|ru)$`),
		RE(`^(?:перевод|translation)${SEP}(?:ru|russian|на${SEP}русский)$`),
		RE(`^(?:орысша|русский|russian|ru)${SEP}(?:аударма|перевод|translation)$`),
	],
	translation_en: [
		RE(`^(?:ағылшынша|английский|english|en)$`),
		RE(`^(?:перевод|translation)${SEP}(?:en|english|на${SEP}английский)$`),
		RE(`^(?:ағылшынша|английский|english|en)${SEP}(?:аударма|перевод|translation)$`),
	],
	description_kz: [
		RE(
			`^(?:қазақша${SEP}сипаттама|казахское${SEP}описание|kazakh${SEP}description|description${SEP}kz|kz${SEP}description)$`,
		),
		RE(`^(?:қазақшасипаттамасы|казахскоеописание|kazakhdescription|descriptionkz|kzdescription)$`),
	],
	description_ru: [
		RE(
			`^(?:орысша${SEP}сипаттама|русское${SEP}описание|russian${SEP}description|description${SEP}ru|ru${SEP}description)$`,
		),
		RE(`^(?:орысшасипаттамасы|русскоеописание|russiandescription|descriptionru|rudescription)$`),
	],
	description_en: [
		RE(
			`^(?:ағылшынша${SEP}сипаттама|английское${SEP}описание|english${SEP}description|description${SEP}en|en${SEP}description)$`,
		),
		RE(
			`^(?:ағылшыншасипаттама|английскоеописание|englishdescription|descriptionen|endescription)$`,
		),
	],
	module_id: [RE(`^(?:module${SEP}id|module|ид${SEP}модуля|модуль)$`), RE(`^(?:moduleid)$`)],
} as const

export type HeaderKind = keyof typeof HEADER_PATTERNS

export function matchHeaderKind(rawHeader: string, kind: HeaderKind): boolean {
	const h = String(rawHeader ?? '').trim()
	return HEADER_PATTERNS[kind].some(re => re.test(h))
}

export function detectVocabHeader(rawHeader: string): HeaderKind | undefined {
	const kinds = Object.keys(HEADER_PATTERNS) as HeaderKind[]
	return kinds.find(k => matchHeaderKind(rawHeader, k))
}

/** For sheet detection in parseExcelBuffer */
export function isVocabHeaderRegex(header: unknown): boolean {
	return typeof header === 'string' && !!detectVocabHeader(header)
}

/** Utilities to extract values by header kind (works on raw or normalized keys) */
function firstByKinds(row: Record<string, unknown>, kinds: HeaderKind[]): string | undefined {
	for (const [key, val] of Object.entries(row)) {
		if (!kinds.some(k => matchHeaderKind(key, k))) continue
		if (val == null) continue
		const s = typeof val === 'string' ? val.trim() : typeof val === 'number' ? String(val) : ''
		if (s) return s
	}
	return undefined
}

export const getWord = (row: Record<string, unknown>) => firstByKinds(row, ['word'])
export const getExample = (row: Record<string, unknown>) => firstByKinds(row, ['example'])
export const getTranslationKZ = (row: Record<string, unknown>) =>
	firstByKinds(row, ['translation_kz'])
export const getTranslationRU = (row: Record<string, unknown>) =>
	firstByKinds(row, ['translation_ru'])
export const getTranslationEN = (row: Record<string, unknown>) =>
	firstByKinds(row, ['translation_en'])
export const getDescriptionKZ = (row: Record<string, unknown>) =>
	firstByKinds(row, ['description_kz'])
export const getDescriptionRU = (row: Record<string, unknown>) =>
	firstByKinds(row, ['description_ru'])
export const getDescriptionEN = (row: Record<string, unknown>) =>
	firstByKinds(row, ['description_en'])
export const getModuleIdCell = (row: Record<string, unknown>) => firstByKinds(row, ['module_id'])

export type ExtractedTranslation = {
	language: string
	translation: string
	description?: string
}
