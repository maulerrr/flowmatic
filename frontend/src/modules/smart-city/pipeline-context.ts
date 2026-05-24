import type { InjectionKey, UnwrapNestedRefs } from 'vue'
import type { useSmartCityPipeline } from './composables/useSmartCityPipeline'

export type PipelineWorkbenchContext = UnwrapNestedRefs<ReturnType<typeof useSmartCityPipeline>>

export const PipelineWorkbenchKey: InjectionKey<PipelineWorkbenchContext> = Symbol('pipeline-workbench')
