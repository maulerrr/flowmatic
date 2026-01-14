import { Injectable, Logger } from '@nestjs/common'
import { QdrantClient, Schemas } from '@qdrant/js-client-rest'
import { AppConfigService } from '../config/config.service'

export type Distance = 'Cosine' | 'Euclid' | 'Dot'

export interface VectorPoint<TPayload extends Record<string, unknown> = Record<string, unknown>> {
	id: number | string
	vector: number[]
	payload?: TPayload
}

@Injectable()
export class QdrantAdapter {
	private readonly logger = new Logger(QdrantAdapter.name)
	private client: QdrantClient

	constructor(private readonly config: AppConfigService) {
		const { url, apiKey } = this.config.qdrant
		this.client = new QdrantClient({ url, apiKey })
	}

	async health(): Promise<boolean> {
		try {
			await this.client.getCollections()
			return true
		} catch (err) {
			this.logger.warn(`Qdrant health check failed: ${String(err)}`)
			return false
		}
	}

	async ensureCollection(
		collection: string,
		vectorSize: number,
		distance: Distance = 'Cosine',
	): Promise<void> {
		const res = await this.client.getCollections()
		const exists = (res.collections || []).some(c => c.name === collection)
		if (!exists) {
			await this.client.createCollection(collection, {
				vectors: { size: vectorSize, distance },
			})
			this.logger.log(
				`Created Qdrant collection '${collection}' (size=${vectorSize}, distance=${distance})`,
			)
		}
	}

	async upsert<TPayload extends Record<string, unknown> = Record<string, unknown>>(
		collection: string,
		points: VectorPoint<TPayload>[],
	): Promise<void> {
		if (!points?.length) return
		await this.client.upsert(collection, { points })
	}

	/** Replace points (delete missing ids then insert) */
	async overwrite<TPayload extends Record<string, unknown> = Record<string, unknown>>(
		collection: string,
		points: VectorPoint<TPayload>[],
	): Promise<void> {
		if (!points?.length) return
		const ids = points.map(p => p.id)
		await this.client.delete(collection, { points: ids })
		await this.client.upsert(collection, { points })
	}

	async deleteByIds(collection: string, ids: Array<number | string>): Promise<void> {
		if (!ids?.length) return
		await this.client.delete(collection, { points: ids })
	}

	async deleteByFilter(collection: string, filter: Schemas['Filter']): Promise<void> {
		await this.client.delete(collection, { filter })
	}

	async getByIds<TPayload extends Record<string, unknown> = Record<string, unknown>>(
		collection: string,
		ids: Array<number | string>,
	): Promise<Array<{ id: number | string; vector?: number[]; payload?: TPayload }>> {
		if (!ids?.length) return []
		const res = await this.client.retrieve(collection, {
			ids,
			with_payload: true,
			with_vector: false,
		})
		return res as Array<{ id: number | string; vector?: number[]; payload?: TPayload }>
	}

	async scroll<TPayload extends Record<string, unknown> = Record<string, unknown>>(
		collection: string,
		limit = 100,
		filter?: Schemas['Filter'],
		offset?: number,
	): Promise<{
		points: Array<{ id: number | string; payload?: TPayload }>
		next_page_offset?: number
	}> {
		const res = await this.client.scroll(collection, {
			limit,
			offset,
			filter,
			with_payload: true,
			with_vector: false,
		})
		return {
			points: res.points as Array<{ id: number | string; payload?: TPayload }>,
			next_page_offset: typeof res.next_page_offset === 'number' ? res.next_page_offset : undefined,
		}
	}

	async search<TPayload extends Record<string, unknown> = Record<string, unknown>>(
		collection: string,
		vector: number[],
		limit = 10,
		filter?: Schemas['Filter'],
	) {
		const result = await this.client.search(collection, {
			vector,
			limit,
			filter,
			with_payload: true,
			with_vector: false,
		})
		return result as Array<{
			id: number | string
			score: number
			payload?: TPayload
		}>
	}

	async createPayloadIndex(
		collection: string,
		field: string,
		fieldSchema: Schemas['PayloadSchemaType'] = 'keyword',
	): Promise<void> {
		await this.client.createPayloadIndex(collection, {
			field_name: field,
			field_schema: fieldSchema,
		})
	}

	async dropCollection(collection: string): Promise<void> {
		await this.client.deleteCollection(collection)
	}

	async recreateCollection(
		collection: string,
		vectorSize: number,
		distance: Distance = 'Cosine',
	): Promise<void> {
		try {
			await this.dropCollection(collection)
		} catch {
			// Ignore error if collection doesn't exist
		}
		await this.ensureCollection(collection, vectorSize, distance)
	}
}
