-- CreateTable
CREATE TABLE "PipelineExport" (
    "id" TEXT NOT NULL,
    "pipelineRunId" TEXT NOT NULL,
    "adapterType" TEXT NOT NULL,
    "destination" TEXT NOT NULL,
    "recordsExported" INTEGER NOT NULL DEFAULT 0,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PipelineExport_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PipelineExport_pipelineRunId_idx" ON "PipelineExport"("pipelineRunId");

-- CreateIndex
CREATE INDEX "PipelineExport_adapterType_idx" ON "PipelineExport"("adapterType");

-- CreateIndex
CREATE INDEX "PipelineExport_createdAt_idx" ON "PipelineExport"("createdAt");

-- AddForeignKey
ALTER TABLE "PipelineExport" ADD CONSTRAINT "PipelineExport_pipelineRunId_fkey" FOREIGN KEY ("pipelineRunId") REFERENCES "PipelineRun"("id") ON DELETE CASCADE ON UPDATE CASCADE;
