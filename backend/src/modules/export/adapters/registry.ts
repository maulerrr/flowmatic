import { Injectable, Logger } from '@nestjs/common'
import { ExportAdapter, ExportAdapterType } from '../types/export.types'
import { BaseExportAdapter } from './base.adapter'
import { PostgresExportAdapter } from './postgres.adapter'
import { MongoDBExportAdapter } from './mongodb.adapter'
import { HuggingFaceExportAdapter } from './huggingface.adapter'
import { CSVExportAdapter, JSONExportAdapter } from './file.adapters'
import { StorageService } from '../../storage/storage.service'

/**
 * Registry for managing export adapters
 * Provides factory methods to get adapters by type and list available adapters
 */
@Injectable()
export class ExportAdapterRegistry {
	private readonly logger = new Logger(ExportAdapterRegistry.name)
	private adapters: Map<ExportAdapterType, ExportAdapter> = new Map()

	constructor(private storageService: StorageService) {
		this.registerBuiltInAdapters()
	}

	/**
	 * Register built-in adapters
	 */
	private registerBuiltInAdapters() {
		// Database adapters
		this.register(new PostgresExportAdapter())
		this.register(new MongoDBExportAdapter())

		// Cloud adapters
		this.register(new HuggingFaceExportAdapter())

		// File adapters
		this.register(new CSVExportAdapter(this.storageService))
		this.register(new JSONExportAdapter(this.storageService))

		this.logger.log(`Registered ${this.adapters.size} export adapters`)
	}

	/**
	 * Register a custom adapter
	 */
	register(adapter: ExportAdapter) {
		this.adapters.set(adapter.type, adapter)
		this.logger.log(`Registered adapter: ${adapter.name} (${adapter.type})`)
	}

	/**
	 * Get adapter by type
	 */
	getAdapter(type: ExportAdapterType): ExportAdapter | undefined {
		return this.adapters.get(type)
	}

	/**
	 * Get all registered adapters
	 */
	getAllAdapters(): ExportAdapter[] {
		return Array.from(this.adapters.values())
	}

	/**
	 * Get adapter metadata for UI
	 */
	getAdapterMetadata() {
		return this.getAllAdapters().map(adapter => ({
			type: adapter.type,
			name: adapter.name,
			description: adapter.description,
			requiredSettings: adapter.requiredSettings,
		}))
	}

	/**
	 * Check if adapter type is registered
	 */
	hasAdapter(type: ExportAdapterType): boolean {
		return this.adapters.has(type)
	}
}
