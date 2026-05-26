ALTER TABLE "SmartCityPipeline"
ADD COLUMN "insightIntervalMinutes" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN "insightDepth" TEXT NOT NULL DEFAULT 'standard',
ADD COLUMN "insightFocus" TEXT NOT NULL DEFAULT 'all',
ADD COLUMN "lastInsightRunAt" TIMESTAMP(3);

CREATE TABLE "PipelineInsightRun" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "trigger" TEXT NOT NULL DEFAULT 'scheduled',
    "status" TEXT NOT NULL DEFAULT 'RUNNING',
    "intervalMinutes" INTEGER,
    "depth" TEXT NOT NULL DEFAULT 'standard',
    "focus" TEXT NOT NULL DEFAULT 'all',
    "profileJson" JSONB NOT NULL DEFAULT '{}',
    "findingsJson" JSONB NOT NULL DEFAULT '{}',
    "planJson" JSONB NOT NULL DEFAULT '{}',
    "visualizationsJson" JSONB NOT NULL DEFAULT '[]',
    "narrative" TEXT,
    "errorMessage" TEXT,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PipelineInsightRun_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "PipelineInsightRun_organizationId_idx" ON "PipelineInsightRun"("organizationId");
CREATE INDEX "PipelineInsightRun_pipelineId_createdAt_idx" ON "PipelineInsightRun"("pipelineId", "createdAt");
CREATE INDEX "PipelineInsightRun_status_idx" ON "PipelineInsightRun"("status");

ALTER TABLE "PipelineInsightRun" ADD CONSTRAINT "PipelineInsightRun_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "PipelineInsightRun" ADD CONSTRAINT "PipelineInsightRun_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;
