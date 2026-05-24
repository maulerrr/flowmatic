CREATE TABLE "ModelTrainingRun" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT,
    "createdByUserId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "datasetPath" TEXT NOT NULL,
    "modelType" TEXT NOT NULL DEFAULT 'TRAFFIC_BASELINE',
    "status" TEXT NOT NULL DEFAULT 'QUEUED',
    "localArtifactPath" TEXT,
    "datasetProfile" JSONB NOT NULL DEFAULT '{}',
    "featureSpec" JSONB NOT NULL DEFAULT '{}',
    "metricsJson" JSONB NOT NULL DEFAULT '{}',
    "logsJson" JSONB NOT NULL DEFAULT '[]',
    "errorMessage" TEXT,
    "startedAt" TIMESTAMP(3),
    "finishedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelTrainingRun_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "ModelArtifact" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "trainingRunId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "modelType" TEXT NOT NULL,
    "version" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'LOCAL_ONLY',
    "localPath" TEXT NOT NULL,
    "s3Uri" TEXT,
    "featureSpec" JSONB NOT NULL DEFAULT '{}',
    "metricsJson" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelArtifact_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "ModelTrainingRun_organizationId_idx" ON "ModelTrainingRun"("organizationId");
CREATE INDEX "ModelTrainingRun_pipelineId_idx" ON "ModelTrainingRun"("pipelineId");
CREATE INDEX "ModelTrainingRun_status_idx" ON "ModelTrainingRun"("status");
CREATE INDEX "ModelTrainingRun_createdAt_idx" ON "ModelTrainingRun"("createdAt");

CREATE INDEX "ModelArtifact_organizationId_idx" ON "ModelArtifact"("organizationId");
CREATE INDEX "ModelArtifact_trainingRunId_idx" ON "ModelArtifact"("trainingRunId");
CREATE INDEX "ModelArtifact_status_idx" ON "ModelArtifact"("status");
CREATE INDEX "ModelArtifact_createdAt_idx" ON "ModelArtifact"("createdAt");

ALTER TABLE "ModelTrainingRun" ADD CONSTRAINT "ModelTrainingRun_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ModelArtifact" ADD CONSTRAINT "ModelArtifact_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ModelArtifact" ADD CONSTRAINT "ModelArtifact_trainingRunId_fkey" FOREIGN KEY ("trainingRunId") REFERENCES "ModelTrainingRun"("id") ON DELETE CASCADE ON UPDATE CASCADE;
