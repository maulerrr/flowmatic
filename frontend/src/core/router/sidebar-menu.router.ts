// src/core/router/sidebar-menu.router.ts
import {
	BookText,
	ChartNoAxesGantt,
	ClipboardList,
	Copy,
	Dumbbell,
	Home,
	ImageUp,
	Layers,
	LifeBuoy,
	type LucideIcon,
	Users,
	Wrench,
} from 'lucide-vue-next'

// ----- Nav Item Types -----
export interface PlainNavItem {
	type: 'plain'
	title: string
	url: string
	icon?: LucideIcon
	isActive?: boolean
}

export interface CollapsibleNavItem {
	type: 'collapsible'
	title: string
	icon?: LucideIcon
	isActive?: boolean
	items: Array<{
		title: string
		url: string
		isActive?: boolean
	}>
}

export type NavItem = PlainNavItem | CollapsibleNavItem

export interface SidebarGroup {
	label: string
	items: NavItem[]
}

// ----- Grouped Menu -----
export const SIDEBAR_GROUPS: SidebarGroup[] = [
	{
		label: 'Жүйені басқару', // “System”
		items: [
			{ type: 'plain', title: 'Басқару тақтасы', url: '/admin', icon: Home },
			{ type: 'plain', title: 'Қолдау көрсету', url: '/support', icon: LifeBuoy },
		],
	},
	{
		label: 'Пайдаланушылар', // “Users”
		items: [
			{
				type: 'collapsible',
				title: 'Пайдаланушылар',
				icon: Users,
				items: [
					{ title: 'Оқушылар', url: '/admin/users/learners' },
					{ title: 'Администраторлар', url: '/admin/users/admins' },
					{ title: 'Мұғалімдер', url: '/admin/users/teachers' },
					{ title: 'Қолдау қызметі', url: '/admin/users/supports' },
				],
			},
			// { type: 'plain', title: 'Сессиялар', url: '/admin/sessions', icon: LogOut },
		],
	},
	{
		label: 'Контент', // “Content”
		items: [
			{ type: 'plain', title: 'Деңгейлер', url: '/admin/levels', icon: ChartNoAxesGantt },
			{ type: 'plain', title: 'Модульдер', url: '/admin/modules', icon: Layers },
			{
				type: 'plain',
				title: 'Барлық негізгі жаттығулар',
				url: '/admin/base-exercises',
				icon: Dumbbell,
			},
			{
				type: 'plain',
				title: 'Флэшкарт суреттерін жүктеу',
				url: '/admin/base-exercises/import-flashcard-images',
				icon: ImageUp,
			},
			{ type: 'plain', title: 'Бағалау сынақтары', url: '/admin/assessments', icon: ClipboardList },
			{
				type: 'plain',
				title: 'Жаттығуларды жөндеу',
				url: '/admin/exercises/repair',
				icon: Wrench,
			},
			{
				type: 'plain',
				title: 'Interest сегменттерін көшіру',
				url: '/admin/interest-segments/recopy',
				icon: Copy,
			},

			{ type: 'plain', title: 'Білім базасы', url: '/admin/knowledge-base', icon: BookText },

			// { type: 'plain', title: 'Сегменттер', url: '/admin/segments', icon: Puzzle },
			// { type: 'plain', title: 'Жаттығулар', url: '/admin/exercises', icon: ClipboardList },
		],
	},
	// {
	// 	label: 'AI және чат', // “AI & Chat”
	// 	items: [
	// 		{ type: 'plain', title: 'Чат журналдары', url: '/admin/chat-logs', icon: MessageCircle },
	// 	],
	// },
]
