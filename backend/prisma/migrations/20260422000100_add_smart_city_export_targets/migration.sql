-- CreateTable
CREATE TABLE "SmartCityExportTarget" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "createdByUserId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "stage" TEXT NOT NULL,
    "adapterType" TEXT NOT NULL,
    "settingsJson" JSONB NOT NULL DEFAULT '{}',
    "saveCredentials" BOOLEAN NOT NULL DEFAULT false,
    "isContinuous" BOOLEAN NOT NULL DEFAULT false,
    "status" TEXT NOT NULL DEFAULT 'ACTIVE',
    "cadenceSeconds" INTEGER NOT NULL DEFAULT 60,
    "lastCursorAt" TIMESTAMP(3),
    "lastRunAt" TIMESTAMP(3),
    "lastRunId" TEXT,
    "lastError" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SmartCityExportTarget_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SmartCityExportRun" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "targetId" TEXT,
    "stage" TEXT NOT NULL,
    "adapterType" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'QUEUED',
    "rowCount" INTEGER NOT NULL DEFAULT 0,
    "recordsExported" INTEGER NOT NULL DEFAULT 0,
    "destination" TEXT,
    "message" TEXT,
    "errorMessage" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "startedAt" TIMESTAMP(3),
    "finishedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SmartCityExportRun_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "SmartCityExportTarget_organizationId_idx" ON "SmartCityExportTarget"("organizationId");

-- CreateIndex
CREATE INDEX "SmartCityExportTarget_pipelineId_idx" ON "SmartCityExportTarget"("pipelineId");

-- CreateIndex
CREATE INDEX "SmartCityExportTarget_status_idx" ON "SmartCityExportTarget"("status");

-- CreateIndex
CREATE INDEX "SmartCityExportTarget_isContinuous_idx" ON "SmartCityExportTarget"("isContinuous");

-- CreateIndex
CREATE INDEX "SmartCityExportRun_organizationId_idx" ON "SmartCityExportRun"("organizationId");

-- CreateIndex
CREATE INDEX "SmartCityExportRun_pipelineId_idx" ON "SmartCityExportRun"("pipelineId");

-- CreateIndex
CREATE INDEX "SmartCityExportRun_targetId_idx" ON "SmartCityExportRun"("targetId");

-- CreateIndex
CREATE INDEX "SmartCityExportRun_status_idx" ON "SmartCityExportRun"("status");

-- CreateIndex
CREATE INDEX "SmartCityExportRun_createdAt_idx" ON "SmartCityExportRun"("createdAt");

-- AddForeignKey
ALTER TABLE "SmartCityExportTarget" ADD CONSTRAINT "SmartCityExportTarget_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SmartCityExportTarget" ADD CONSTRAINT "SmartCityExportTarget_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SmartCityExportRun" ADD CONSTRAINT "SmartCityExportRun_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SmartCityExportRun" ADD CONSTRAINT "SmartCityExportRun_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SmartCityExportRun" ADD CONSTRAINT "SmartCityExportRun_targetId_fkey" FOREIGN KEY ("targetId") REFERENCES "SmartCityExportTarget"("id") ON DELETE SET NULL ON UPDATE CASCADE;
