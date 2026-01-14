/* eslint-disable */
// This file is used to tell TypeScript how to handle .vue files.
// Without this, TypeScript will complain about not being able to find modules ending in .vue.
declare module '*.vue' {
	import type { DefineComponent } from 'vue'
	// DefineComponent is used here to correctly infer the component's props, emitted events, etc.
	const component: DefineComponent<{}, {}, any>
	export default component
}
