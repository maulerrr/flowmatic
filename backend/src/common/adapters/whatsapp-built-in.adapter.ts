import {
	Injectable,
	Logger,
	BadRequestException,
	InternalServerErrorException,
} from '@nestjs/common'
import { HttpService } from '@nestjs/axios'
import { firstValueFrom } from 'rxjs'
import { AxiosError, isAxiosError } from 'axios'
import { AppConfigService } from 'src/common/config/config.service'
import { toWhatsAppNumber } from 'src/common/utils/phone-formatter'

@Injectable()
export class WhatsAppAdapter {
	private readonly logger = new Logger(WhatsAppAdapter.name)
	private readonly url: string
	private readonly token: string

	constructor(
		private readonly http: HttpService,
		private readonly config: AppConfigService,
	) {
		const base = 'https://graph.facebook.com'
		const version = this.config.whatsapp.version
		const phoneId = this.config.whatsapp.phoneId
		this.token = this.config.whatsapp.accessToken

		if (!version || !phoneId || !this.token) {
			this.logger.error('WhatsApp API config missing (VERSION, PHONE_NUMBER_ID or ACCESS_TOKEN)')
			throw new InternalServerErrorException('WhatsApp provider not configured')
		}

		this.url = `${base}/${version}/${phoneId}/messages`
	}

	/**
	 * Send a template‐based OTP message using the hardcoded English template.
	 */
	async sendOtp(toPhone: string, code: string): Promise<void> {
		// temporarily massage +7 numbers into +78…
		const formattedPhone = toWhatsAppNumber(toPhone)

		const payload = {
			messaging_product: 'whatsapp',
			to: formattedPhone,
			type: 'template',
			template: {
				name: 'otp_ru_v1', // hardcoded RU template name
				language: { code: 'ru' }, // must match template locale in WABA (e.g., 'ru' or 'en_US')
				components: [
					{
						type: 'body',
						parameters: [{ type: 'text', text: code }],
					},
					{
						type: 'button',
						sub_type: 'url',
						index: '0',
						parameters: [{ type: 'text', text: code }],
					},
				],
			},
		}

		try {
			const response = await firstValueFrom(
				this.http.post(this.url, payload, {
					headers: {
						Authorization: `Bearer ${this.token}`,
						'Content-Type': 'application/json',
					},
				}),
			)
			this.logger.log(`OTP template sent to ${toPhone} — status ${response.status}`)
		} catch (_err: unknown) {
			if (isAxiosError(_err)) {
				const axiosErr = _err as AxiosError<{ error: { message: string } }>
				const status = axiosErr.response?.status
				const data = axiosErr.response?.data

				this.logger.error(`WhatsApp sendOtp failed [${status}]`, data)

				if (status && status >= 400 && status < 500) {
					throw new BadRequestException(
						data?.error?.message || `Failed to send WhatsApp template (status ${status})`,
					)
				}
			} else {
				this.logger.error('WhatsApp sendOtp failed with non‐Axios error:', String(_err))
			}

			throw new InternalServerErrorException('Failed to send OTP')
		}
	}
}
