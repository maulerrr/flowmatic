import {
	buildDatasetDescription,
	groupRowsByHour,
	hourKeyFromDate,
	hourlyPartPath,
	mergeRowsByEventId,
	updateManifestPart,
	createEmptyHourlyManifest,
} from '../hourly-csv-partition.util'

describe('hourly-csv-partition.util', () => {
	it('groups rows by UTC hour from eventTime', () => {
		const groups = groupRowsByHour([
			{ eventId: 'a', eventTime: '2026-05-21T14:10:00.000Z' },
			{ eventId: 'b', eventTime: '2026-05-21T14:55:00.000Z' },
			{ eventId: 'c', eventTime: '2026-05-21T15:05:00.000Z' },
		])

		expect(groups.size).toBe(2)
		expect(groups.get('2026-05-21T14')).toHaveLength(2)
		expect(groups.get('2026-05-21T15')).toHaveLength(1)
	})

	it('builds stable hourly file paths', () => {
		expect(hourKeyFromDate(new Date('2026-05-21T14:59:59.000Z'))).toBe('2026-05-21T14')
		expect(hourlyPartPath('2026-05-21T14')).toBe('data/hourly/2026-05-21T14.csv')
	})

	it('deduplicates rows by eventId when merging', () => {
		const merged = mergeRowsByEventId(
			[{ eventId: '1', value: 'old' }],
			[
				{ eventId: '1', value: 'duplicate' },
				{ eventId: '2', value: 'new' },
			],
		)
		expect(merged).toHaveLength(2)
		expect(merged[1]).toEqual({ eventId: '2', value: 'new' })
	})

	it('updates manifest totals and description', () => {
		let manifest = createEmptyHourlyManifest()
		manifest = updateManifestPart(manifest, {
			hourKey: '2026-05-21T14',
			path: 'data/hourly/2026-05-21T14.csv',
			newRows: 2,
			totalRows: 10,
		})
		expect(manifest.totalRows).toBe(10)
		expect(buildDatasetDescription(manifest)).toContain('hourly UTC CSV partitions')
	})
})
