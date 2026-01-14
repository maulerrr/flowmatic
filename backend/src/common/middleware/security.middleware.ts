import { Injectable, NestMiddleware } from '@nestjs/common'
import { Request, Response, NextFunction } from 'express'
import { AppConfigService } from '../config/config.service'

@Injectable()
export class SecurityMiddleware implements NestMiddleware {
	constructor(private readonly appConfigService: AppConfigService) {}

	use(req: Request, res: Response, next: NextFunction) {
		const allowedHosts = this.appConfigService.security.allowedHosts
		const host = req.headers.host?.split(':')[0] // Remove port if present

		if (allowedHosts.length > 0 && host && !allowedHosts.includes(host)) {
			return res.status(400).send('Bad Request: Invalid Host Header')
		}
		next()
	}
}
