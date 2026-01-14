<script setup lang="ts">
import { Bell, ChevronsUpDown, LogOut, Settings2 } from 'lucide-vue-next';
import { useRouter } from 'vue-router';



import { Avatar, AvatarFallback, AvatarImage } from '@/core/components/ui/avatar';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/core/components/ui/dropdown-menu';
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem, useSidebar } from '@/core/components/ui/sidebar';
import { apiClient } from '@/core/configs/axios-instance.config';



import type { AdminDto } from "@/modules/auth/models/auth.models";
import { authService } from '@/modules/auth/services/auth.service';





defineProps<{
  user: AdminDto
}>()

const router = useRouter();

const handleLogout = async () => {
  try {
    await authService.logout();
  } catch (e) {
    console.warn('Logout API call failed', e);
  } finally {
    // Always clear local state even if API fails
    localStorage.removeItem('authTokens');
    delete apiClient.defaults.headers.common['Authorization'];
    router.push('/');
  }
};
</script>

<template>
	<SidebarMenu class="w-fit">
		<SidebarMenuItem>
			<DropdownMenu>
				<DropdownMenuTrigger as-child>
					<SidebarMenuButton
						size="lg"
						class="bg-white data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground cursor-pointer"
					>
						<Avatar class="rounded-lg w-8 h-8">
							<AvatarImage
								:src="user.name"
								:alt="user.name"
							/>
							<AvatarFallback class="rounded-lg">
								{{ user.name.charAt(0) || 'U' }}
							</AvatarFallback>
						</Avatar>
						<div class="flex-1 grid ml-2 text-sm text-left leading-tight">
							<span class="font-semibold truncate">{{ user.name }}</span>
							<span class="text-xs truncate">{{ user.phoneNumber }}</span>
						</div>
						<ChevronsUpDown class="ml-4 size-4" />
					</SidebarMenuButton>
				</DropdownMenuTrigger>

				<DropdownMenuContent
					class="rounded-lg w-[--reka-dropdown-menu-trigger-width] min-w-56"
					side="bottom"
					align="end"
					:side-offset="4"
				>
					<!-- User Info -->
					<DropdownMenuLabel class="p-0 font-normal">
						<div class="flex items-center gap-2 px-1 py-1.5 text-sm">
							<Avatar class="rounded-lg w-8 h-8">
								<AvatarImage
									:src="user.name"
									:alt="user.name"
								/>
								<AvatarFallback class="rounded-lg">
									{{ user.name.charAt(0) || 'U' }}
								</AvatarFallback>
							</Avatar>
							<div class="flex-1 grid text-sm leading-tight">
								<span class="font-semibold truncate">{{ user.name }}</span>
								<span class="text-xs truncate">{{ user.phoneNumber }}</span>
							</div>
						</div>
					</DropdownMenuLabel>

					<DropdownMenuSeparator />

					<!-- Action Items -->
					<DropdownMenuItem>
						<Bell />
						<span>Хабарламалар</span>
					</DropdownMenuItem>
					<DropdownMenuItem>
						<Settings2 />
						<span>Профиль баптаулары</span>
					</DropdownMenuItem>

					<DropdownMenuSeparator />

					<DropdownMenuItem @click="handleLogout">
						<LogOut />
						<span>Шығу</span>
					</DropdownMenuItem>
				</DropdownMenuContent>
			</DropdownMenu>
		</SidebarMenuItem>
	</SidebarMenu>
</template>
