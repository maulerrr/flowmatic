import { useDark, useToggle } from '@vueuse/core'

/**
 * Reactive Dark Mode state.
 * Uses 'theme' key in localStorage and toggles 'dark' class on html element.
 */
export const isDark = useDark({
	selector: 'html',
	attribute: 'class',
	valueDark: 'dark',
	valueLight: '',
	storageKey: 'theme', // Matches existing key
})

export const toggleDark = useToggle(isDark)
