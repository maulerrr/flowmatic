import { parseComposerMessage, resolveSlashCommand } from '../pipeline-copilot.commands'

describe('pipeline-copilot.commands', () => {
	it('parses slash command with leading question text', () => {
		const parsed = parseComposerMessage('What is going on in this pipeline? /funnel')
		expect(parsed.questionText).toBe('What is going on in this pipeline?')
		expect(parsed.vizChipIds).toEqual(['data_funnel'])
	})

	it('parses slash-only chart request', () => {
		const parsed = parseComposerMessage('/event-volume')
		expect(parsed.questionText).toBe('')
		expect(parsed.vizChipIds).toEqual(['event_volume'])
	})

	it('strips slash tokens from LLM question text', () => {
		const parsed = parseComposerMessage('Explain exports /export-health today')
		expect(parsed.questionText).toBe('Explain exports today')
		expect(parsed.vizChipIds).toEqual(['export_health'])
	})

	it('resolves slash aliases', () => {
		expect(resolveSlashCommand('funnel')).toBe('data_funnel')
		expect(resolveSlashCommand('events')).toBe('event_volume')
	})
})
