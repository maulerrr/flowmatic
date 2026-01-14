import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import path from 'node:path'
import { defineConfig } from 'vite'
import viteCompression from 'vite-plugin-compression'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		vue(),
		vueDevTools(),
		tailwindcss(),
		viteCompression({
			algorithm: 'gzip',
			ext: '.gz',
			threshold: 10240,
		}),
		viteCompression({
			algorithm: 'brotliCompress',
			ext: '.br',
			threshold: 10240,
		}),
	],
	resolve: {
		alias: {
			'@': path.resolve(__dirname, './src'),
			'lucide-vue-next': 'lucide-vue-next/dist/esm/lucide-vue-next.js',
		},
	},
	css: {
		preprocessorOptions: {
			scss: {
				api: 'modern-compiler',
			},
		},
	},
	optimizeDeps: {
		include: ['lucide-vue-next', 'quill', '@vueup/vue-quill', 'vue-quill-editor'],
	},
	build: {
		sourcemap: false,
		commonjsOptions: {
			include: [/node_modules/],
			transformMixedEsModules: true,
		},
	},
	ssr: {
		noExternal: ['@vueup/vue-quill'],
	},
})
