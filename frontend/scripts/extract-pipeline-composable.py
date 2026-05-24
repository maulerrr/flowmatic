from pathlib import Path

src_path = Path(__file__).resolve().parents[1] / "src/pages/connectors-page.vue"
src = src_path.read_text(encoding="utf-8")
marker = '<script setup lang="ts">'
start = src.index(marker) + len(marker)
end = src.index("</script>")
body = src[start:end].strip()
body = body.replace("import '@/modules/smart-city/styles/pipeline-workbench.css'\n\n", "")

header = """import {
\tActivity,
\tBrain,
\tCpu,
\tDatabase,
\tHardDrive,
\tLayoutDashboard,
\tNetwork,
\tPlus,
\tRadio,
\tRefreshCw,
\tRouter,
\tSave,
\tTrash2,
\tVideo,
\tWifi,
\tWorkflow,
\tX,
\tZap,
} from 'lucide-vue-next'
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { toast } from 'vue-sonner'
import {
\tapiClient,
\ttype SmartCityBackfillResult,
\ttype SmartCityObservability,
\ttype DataLakeConnection,
\ttype DataLakeObjectGroup,
\ttype ExportAdapterMetadata,
\ttype FederatedRound,
\ttype SmartCityExportRun,
\ttype SmartCityExportTarget,
\ttype FederatedConnectionState,
\ttype ModelArtifact,
\ttype ModelTrainingRun,
\ttype ResearchModel,
\ttype SensorEvent,
\ttype SensorSource,
\ttype SensorSourceTestResult,
\ttype SmartCityPipeline,
} from '@/api/client'

export type PipelineStage = 'sources' | 'processing' | 'lake' | 'federated'
export type ExportStage = 'raw' | 'cleaned' | 'business'
export type ExportAdapterKind = 'json' | 'csv' | 'postgres' | 'mongodb' | 'huggingface'

export function useSmartCityPipeline() {
"""

footer = """
\treturn {
\t\tviewMode,
\t\tactiveStage,
\t\tloading,
\t\tsaving,
\t\tpipelines,
\t\tselectedPipelineId,
\t\tsources,
\t\tevents,
\t\tdataLakes,
\t\texportAdapters,
\t\texportTargets,
\t\texportRuns,
\t\tdataLakeObjects,
\t\tlatestBackfill,
\t\tfederatedRounds,
\t\tobservability,
\t\tmodels,
\t\ttrainingRuns,
\t\tresearchModels,
\t\tlatestProcessingResult,
\t\tn8nWorkflow,
\t\tlogs,
\t\twsStatus,
\t\tsimulatorPresets,
\t\tdemoFederatedEndpoint,
\t\tdemoSimulatorBaseUrl,
\t\tshowPipelineModal,
\t\tshowSourceModal,
\t\tshowConfigModal,
\t\tshowDataLakeModal,
\t\tshowAdvancedExportJson,
\t\tpipelineForm,
\t\tsourceForm,
\t\tdataLakeForm,
\t\texportForm,
\t\tbackfillForm,
\t\texportSettingsForm,
\t\tstreamConfig,
\t\tfederatedForm,
\t\tfederatedRoundForm,
\t\tfederatedUpdateForm,
\t\tfederatedAggregateForm,
\t\tselectedPipeline,
\t\trunningSourceCount,
\t\tactiveDataLake,
\t\tthroughput,
\t\tsourceStatusText,
\t\trecentEventsPreview,
\t\tactiveResearchModel,
\t\tactiveDbModel,
\t\tactiveModelLabel,
\t\tlatestTrainingRun,
\t\tprocessingModelCount,
\t\tlinkedDataLake,
\t\tselectedExportAdapter,
\t\texportCredentialHint,
\t\texportSettingsPreview,
\t\tsupportsSavedCredentials,
\t\tfederatedConfig,
\t\tactiveFederatedRound,
\t\tsourceMixSummary,
\t\tsimulatorEndpointHint,
\t\tloadPipelines,
\t\tloadDashboard,
\t\trefreshEvents,
\t\tloadN8nWorkflow,
\t\tloadModels,
\t\tloadFederatedRounds,
\t\tloadObservability,
\t\tcreatePipeline,
\t\tarchivePipeline,
\t\taddSource,
\t\tstartSource,
\t\tstopSource,
\t\ttestSource,
\t\tremoveSource,
\t\tsaveConfiguration,
\t\tsaveDataLake,
\t\ttestDataLake,
\t\tdisconnectDataLake,
\t\tdeleteDataLakeConnection,
\t\tpollHttpFeed,
\t\ttrainAstanaModel,
\t\tpromoteModel,
\t\tdeployModel,
\t\tdeployResearchModel,
\t\ttestProcessingUnit,
\t\texportStageToAdapter,
\t\tsaveExportTarget,
\t\trunExportTarget,
\t\trefreshLakeBrowser,
\t\tbackfillDataLakeStages,
\t\tformatBytes,
\t\ttoggleContinuousTarget,
\t\tconnectFederated,
\t\ttestFederatedConnection,
\t\tdisconnectFederatedConnection,
\t\tstartFederatedRoundFlow,
\t\tsubmitFederatedRoundUpdate,
\t\taggregateFederatedRoundFlow,
\t\tsyncFederatedGlobalState,
\t\topenFederatedWorkspace,
\t\tloadExportTargetIntoForm,
\t\taddLog,
\t\tresetSourceForm,
\t\thandleSourceTestResult,
\t\tsourceIcon,
\t\tformatDateTime,
\t}
}
"""

indented = "\n".join(("\t" + line if line.strip() else line) for line in body.splitlines())
out = Path(__file__).resolve().parents[1] / "src/modules/smart-city/composables/useSmartCityPipeline.ts"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(header + indented + footer, encoding="utf-8")
print(f"wrote {out}")
