import {
	Brain,
	Cpu,
	HardDrive,
	Radio,
	Router,
	Video,
	Wifi,
	Zap,
	type LucideIcon,
} from 'lucide-vue-next'

export function formatBytes(value: number): string {
	if (!Number.isFinite(value) || value <= 0) return '0 B'
	const units = ['B', 'KB', 'MB', 'GB', 'TB']
	const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
	const scaled = value / 1024 ** index
	return `${scaled.toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatDateTime(value?: string | null): string {
	if (!value) return '—'
	const date = new Date(value)
	if (Number.isNaN(date.getTime())) return value
	return date.toLocaleString()
}

export function sourceIcon(kind: string): LucideIcon {
	switch (kind) {
		case 'traffic':
			return Router
		case 'air':
			return Zap
		case 'energy':
			return Cpu
		case 'video':
			return Video
		case 'wifi':
			return Wifi
		case 'brain':
			return Brain
		case 'storage':
			return HardDrive
		default:
			return Radio
	}
}
