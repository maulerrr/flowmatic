import { Body, Controller, Delete, Get, Param, Put, Query, Req, UseGuards } from '@nestjs/common'
import { ApiTags } from '@nestjs/swagger'
import { AuthGuard } from '../auth/auth.guard'
import { AuthenticatedRequest } from 'src/common/types/http.types'
import { DeployHuggingFaceModelDto } from './dto/deploy-huggingface-model.dto'
import { ListHuggingFaceModelsQueryDto } from './dto/list-huggingface-models-query.dto'
import { SaveHuggingFaceTokenDto } from './dto/save-huggingface-token.dto'
import { HuggingFaceIntegrationService } from './huggingface.integration.service'

@ApiTags('integrations')
@Controller('integrations/huggingface')
@UseGuards(AuthGuard)
export class IntegrationsController {
	constructor(private readonly huggingFace: HuggingFaceIntegrationService) {}

	@Get('status')
	async status(@Req() req: AuthenticatedRequest) {
		return {
			success: true,
			data: await this.huggingFace.getStatus(req.authContext!.organizationId),
		}
	}

	@Put('token')
	async saveToken(@Req() req: AuthenticatedRequest, @Body() body: SaveHuggingFaceTokenDto) {
		return {
			success: true,
			data: await this.huggingFace.saveToken(req.authContext!.organizationId, body.token),
		}
	}

	@Delete('token')
	async removeToken(@Req() req: AuthenticatedRequest) {
		return {
			success: true,
			data: await this.huggingFace.removeToken(req.authContext!.organizationId),
		}
	}

	@Get('models')
	async listModels(@Req() req: AuthenticatedRequest, @Query() query: ListHuggingFaceModelsQueryDto) {
		return {
			success: true,
			data: await this.huggingFace.listModels(req.authContext!.organizationId, query),
		}
	}

	@Get('models/:modelId')
	async getModel(@Req() req: AuthenticatedRequest, @Param('modelId') modelId: string) {
		return {
			success: true,
			data: await this.huggingFace.getModelDetails(
				req.authContext!.organizationId,
				decodeURIComponent(modelId),
			),
		}
	}
}
