import { Module } from '@nestjs/common'
import { AppConfigModule } from './config/config.module'
import { MediaUrlService } from './services/media-url.service'
import { QdrantAdapter } from './adapters/qdrant.adapter'

@Module({
	imports: [AppConfigModule],
	providers: [MediaUrlService, QdrantAdapter],
	exports: [MediaUrlService, QdrantAdapter],
})
export class CommonModule {}
