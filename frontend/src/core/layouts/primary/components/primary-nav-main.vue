<script setup lang="ts">
import { ChevronRight } from 'lucide-vue-next';
import { RouterLink } from 'vue-router';



import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/core/components/ui/collapsible';
import { SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem } from '@/core/components/ui/sidebar';
import type { CollapsibleNavItem, NavItem } from '@/core/router/sidebar-menu.router';





const props = defineProps<{
  label: string
  items: NavItem[]
}>()
</script>

<template>
	<SidebarGroup>
		<SidebarGroupLabel>{{ props.label }}</SidebarGroupLabel>
		<SidebarMenu>
			<template
				v-for="item in props.items"
				:key="item.title"
			>
				<!-- Plain link -->
				<SidebarMenuItem v-if="item.type === 'plain'">
					<SidebarMenuButton
						:isActive="item.isActive"
						:tooltip="item.title"
						as-child
					>
						<RouterLink :to="item.url">
							<component
								:is="item.icon"
								v-if="item.icon"
							/>
							<span>{{ item.title }}</span>
						</RouterLink>
					</SidebarMenuButton>
				</SidebarMenuItem>

				<!-- Collapsible group -->
				<Collapsible
					v-else-if="item.type === 'collapsible'"
					as-child
					:default-open="item.isActive"
					class="group/collapsible"
				>
					<SidebarMenuItem>
						<CollapsibleTrigger as-child>
							<SidebarMenuButton
								:isActive="item.isActive"
								:tooltip="item.title"
							>
								<component
									:is="item.icon"
									v-if="item.icon"
								/>
								<span>{{ item.title }}</span>
								<ChevronRight
									class="ml-auto group-data-[state=open]/collapsible:rotate-90 transition-transform duration-200"
								/>
							</SidebarMenuButton>
						</CollapsibleTrigger>
						<CollapsibleContent>
							<SidebarMenuSub>
								<SidebarMenuSubItem
									v-for="sub in (item as CollapsibleNavItem).items"
									:key="sub.title"
								>
									<SidebarMenuSubButton
										:isActive="sub.isActive"
										as-child
									>
										<RouterLink :to="sub.url">
											<span>{{ sub.title }}</span>
										</RouterLink>
									</SidebarMenuSubButton>
								</SidebarMenuSubItem>
							</SidebarMenuSub>
						</CollapsibleContent>
					</SidebarMenuItem>
				</Collapsible>
			</template>
		</SidebarMenu>
	</SidebarGroup>
</template>
