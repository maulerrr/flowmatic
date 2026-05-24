CREATE TABLE "ExportCredential" (
    "id" TEXT NOT NULL,
    "organizationId" TEXT NOT NULL,
    "adapterType" TEXT NOT NULL,
    "encryptedSettings" TEXT NOT NULL,
    "iv" TEXT NOT NULL,
    "authTag" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ExportCredential_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "ExportCredential_organizationId_adapterType_key" ON "ExportCredential"("organizationId", "adapterType");
CREATE INDEX "ExportCredential_organizationId_idx" ON "ExportCredential"("organizationId");
CREATE INDEX "ExportCredential_adapterType_idx" ON "ExportCredential"("adapterType");

ALTER TABLE "ExportCredential" ADD CONSTRAINT "ExportCredential_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
