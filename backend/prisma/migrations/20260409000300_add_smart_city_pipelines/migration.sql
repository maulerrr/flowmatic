-- Real-time smart city pipeline primitives.

CREATE TABLE "SmartCityPipeline" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "createdByUserId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "status" TEXT NOT NULL DEFAULT 'DRAFT',
    "graphJson" JSONB NOT NULL DEFAULT '{}',
    "streamConfig" JSONB NOT NULL DEFAULT '{}',
    "activeModelId" TEXT,
    "dataLakeConnectionId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SmartCityPipeline_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "SensorSource" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL DEFAULT 'HTTP_POLLING',
    "sensorKind" TEXT NOT NULL DEFAULT 'iot',
    "mode" TEXT NOT NULL DEFAULT 'SIMULATED',
    "status" TEXT NOT NULL DEFAULT 'STOPPED',
    "endpoint" TEXT,
    "schemaJson" JSONB NOT NULL DEFAULT '{}',
    "connectionConfig" JSONB NOT NULL DEFAULT '{}',
    "pollIntervalMs" INTEGER NOT NULL DEFAULT 5000,
    "lastSeenAt" TIMESTAMP(3),
    "lastError" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "SensorSource_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "SensorEvent" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "pipelineId" TEXT NOT NULL,
    "sourceId" TEXT NOT NULL,
    "eventTime" TIMESTAMP(3) NOT NULL,
    "sensorType" TEXT NOT NULL,
    "location" TEXT,
    "payloadJson" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SensorEvent_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "DataLakeConnection" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT 'CUSTOM_S3',
    "bucket" TEXT NOT NULL,
    "region" TEXT,
    "endpoint" TEXT,
    "basePrefix" TEXT NOT NULL DEFAULT '',
    "accessKeyEncrypted" TEXT,
    "secretKeyEncrypted" TEXT,
    "isDefault" BOOLEAN NOT NULL DEFAULT false,
    "status" TEXT NOT NULL DEFAULT 'CONNECTED',
    "lastTestedAt" TIMESTAMP(3),
    "lastTestStatus" TEXT,
    "pathRulesJson" JSONB NOT NULL DEFAULT '{}',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "DataLakeConnection_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "SmartCityPipeline_organizationId_idx" ON "SmartCityPipeline"("organizationId");
CREATE INDEX "SmartCityPipeline_status_idx" ON "SmartCityPipeline"("status");
CREATE INDEX "SmartCityPipeline_createdAt_idx" ON "SmartCityPipeline"("createdAt");

CREATE INDEX "SensorSource_organizationId_idx" ON "SensorSource"("organizationId");
CREATE INDEX "SensorSource_pipelineId_idx" ON "SensorSource"("pipelineId");
CREATE INDEX "SensorSource_status_idx" ON "SensorSource"("status");
CREATE INDEX "SensorSource_type_idx" ON "SensorSource"("type");

CREATE INDEX "SensorEvent_organizationId_idx" ON "SensorEvent"("organizationId");
CREATE INDEX "SensorEvent_pipelineId_idx" ON "SensorEvent"("pipelineId");
CREATE INDEX "SensorEvent_sourceId_idx" ON "SensorEvent"("sourceId");
CREATE INDEX "SensorEvent_eventTime_idx" ON "SensorEvent"("eventTime");
CREATE INDEX "SensorEvent_createdAt_idx" ON "SensorEvent"("createdAt");

CREATE INDEX "DataLakeConnection_organizationId_idx" ON "DataLakeConnection"("organizationId");
CREATE INDEX "DataLakeConnection_status_idx" ON "DataLakeConnection"("status");
CREATE INDEX "DataLakeConnection_isDefault_idx" ON "DataLakeConnection"("isDefault");

ALTER TABLE "SmartCityPipeline" ADD CONSTRAINT "SmartCityPipeline_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SensorSource" ADD CONSTRAINT "SensorSource_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SensorSource" ADD CONSTRAINT "SensorSource_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SensorEvent" ADD CONSTRAINT "SensorEvent_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SensorEvent" ADD CONSTRAINT "SensorEvent_pipelineId_fkey" FOREIGN KEY ("pipelineId") REFERENCES "SmartCityPipeline"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SensorEvent" ADD CONSTRAINT "SensorEvent_sourceId_fkey" FOREIGN KEY ("sourceId") REFERENCES "SensorSource"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "DataLakeConnection" ADD CONSTRAINT "DataLakeConnection_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
