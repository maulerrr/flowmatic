import 'module-alias/register'

import { ValidationPipe } from '@nestjs/common'
import { NestFactory } from '@nestjs/core'
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger'
import cookieParser from 'cookie-parser'
import helmet from 'helmet'
import morgan from 'morgan'
import { Logger } from 'nestjs-pino'
import { AppModule } from './app.module'
import { AppConfigService } from './common/config/config.service'
import { generationTimeoutMiddleware } from './common/middleware/generation-timeout.middleware'

async function bootstrap() {
	const app = await NestFactory.create(AppModule)
	app.setGlobalPrefix('api/v1')

	app.use(cookieParser())

	const config = new DocumentBuilder()
		.setTitle('Flowmatic API')
		.setDescription('Intelligent Data Preparation Platform API')
		.setVersion('1.0')
		.addTag('flowmatic')
		.build()
	const documentFactory = () => SwaggerModule.createDocument(app, config)
	SwaggerModule.setup('api', app, documentFactory)

	app.useGlobalPipes(
		new ValidationPipe({
			whitelist: true,
			forbidNonWhitelisted: true,
			transform: true,
			transformOptions: {
				enableImplicitConversion: true,
			},
		}),
	)

	app.useLogger(app.get(Logger))
	app.use(morgan('dev'))

	// The SecurityMiddleware handles CORS and Host header validation
	app.enableCors({
		origin: app.get(AppConfigService).security.backendCorsOrigins,
		methods: 'GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS',
		credentials: true,
	})
	app.use(helmet())

	await app.listen(process.env.SERVER_PORT ?? 8080)

	// Set longer timeouts only for long-running generation endpoints (paths containing '/generate').
	// Default: 5 minutes (300000 ms). Can be overridden with REQUEST_TIMEOUT_MS env var.
	try {
		app.use(generationTimeoutMiddleware())
	} catch (err) {
		console.warn('Could not install generation timeout middleware', err)
	}
}

void bootstrap()
