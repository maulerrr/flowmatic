-- CreateTable
CREATE TABLE "PipelineCopilotSession" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "PipelineCopilotSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PipelineCopilotMessage" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "chipId" TEXT,
    "answerSource" TEXT NOT NULL DEFAULT 'rules',
    "visualizationJson" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PipelineCopilotMessage_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PipelineCopilotSession_pipelineId_idx" ON "PipelineCopilotSession"("pipelineId");

-- CreateIndex
CREATE INDEX "PipelineCopilotSession_userId_idx" ON "PipelineCopilotSession"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "PipelineCopilotSession_organizationId_pipelineId_userId_key" ON "PipelineCopilotSession"("organizationId", "pipelineId", "userId");

-- CreateIndex
CREATE INDEX "PipelineCopilotMessage_sessionId_createdAt_idx" ON "PipelineCopilotMessage"("sessionId", "createdAt");

-- AddForeignKey
ALTER TABLE "PipelineCopilotSession" ADD CONSTRAINT "PipelineCopilotSession_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PipelineCopilotSession" ADD CONSTRAINT "PipelineCopilotSession_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PipelineCopilotMessage" ADD CONSTRAINT "PipelineCopilotMessage_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "PipelineCopilotSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;
