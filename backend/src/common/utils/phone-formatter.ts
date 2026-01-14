/**
 * For WhatsApp only: if the number is stored as "+7xxxxxxxxxx",
 * insert an "8" after the "+7" (so it becomes "+78xxxxxxxxxx").
 * If it already starts "+78" or doesn’t start "+7", returns it unchanged.
 */
export function toWhatsAppNumber(phone: string): string {
	if (phone.startsWith('+7') && !phone.startsWith('+78')) {
		return `+7${'8'}${phone.slice(2)}`
	}
	return phone
}

// TODO: fix this issue in WhatsApp Platform, not code issue
