import 'module-alias/register'

import { ValidationPipe } from '@nestjs/common'
import { NestFactory } from '@nestjs/core'
import { FastifyAdapter, NestFastifyApplication } from '@nestjs/platform-fastify'
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger'
import fastifyCookie from '@fastify/cookie'
import fastifyHelmet from '@fastify/helmet'
import fastifyMultipart from '@fastify/multipart'
import fastifyWebsocket from '@fastify/websocket'
import { Logger } from 'nestjs-pino'
import { AppModule } from './app.module'
import { AppConfigService } from './common/config/config.service'
import { registerGenerationTimeoutHook } from './common/middleware/generation-timeout.middleware'
import { registerSecurityHook } from './common/middleware/security.middleware'
import { AuthContextService } from './modules/auth/auth-context.service'
import { SmartCityService } from './modules/smart-city/smart-city.service'

async function bootstrap() {
	const app = await NestFactory.create<NestFastifyApplication>(
		AppModule,
		new FastifyAdapter({ logger: false }),
	)
	app.setGlobalPrefix('api/v1')

	await app.register(fastifyCookie)
	await app.register(fastifyHelmet)
	await app.register(fastifyMultipart, {
		limits: {
			fileSize: 50 * 1024 * 1024,
		},
	})
	await app.register(fastifyWebsocket)

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

	const appConfig = app.get(AppConfigService)
	const fastify = app.getHttpAdapter().getInstance()
	const authContext = app.get(AuthContextService)
	const smartCity = app.get(SmartCityService)

	app.enableCors({
		origin: appConfig.security.backendCorsOrigins,
		methods: 'GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS',
		credentials: true,
	})
	registerSecurityHook(fastify, appConfig.security.allowedHosts)
	fastify.get('/api/v1/smart-city/pipelines/:id/ws', { websocket: true }, (socket: any, request: any) => {
		void (async () => {
			const token = request.cookies?.flowmatic_session
			const context = typeof token === 'string' ? await authContext.validateSession(token) : null
			const pipelineId = request.params?.id

			if (!context || typeof pipelineId !== 'string') {
				socket.close(1008, 'Unauthorized')
				return
			}

			try {
				await smartCity.assertPipelineAccess(context, pipelineId)
				socket.send(JSON.stringify(await smartCity.streamSnapshot(context, pipelineId)))
			} catch {
				socket.close(1008, 'Pipeline not available')
				return
			}

			const unsubscribe = smartCity.subscribeToPipeline(pipelineId, event => {
				if (socket.readyState === socket.OPEN) {
					socket.send(JSON.stringify(event))
				}
			})

			socket.on('message', (raw: Buffer) => {
				const message = raw.toString()
				if (message === 'ping') socket.send(JSON.stringify({ type: 'pong', data: new Date().toISOString() }))
			})
			socket.on('close', unsubscribe)
			socket.on('error', unsubscribe)
		})()
	})

	// Set longer timeouts only for long-running generation endpoints (paths containing '/generate').
	// Default: 5 minutes (300000 ms). Can be overridden with REQUEST_TIMEOUT_MS env var.
	try {
		registerGenerationTimeoutHook(fastify, appConfig.server.requestTimeoutMs)
	} catch (err) {
		console.warn('Could not install generation timeout middleware', err)
	}

	await app.listen(appConfig.server.port)
}

void bootstrap()
