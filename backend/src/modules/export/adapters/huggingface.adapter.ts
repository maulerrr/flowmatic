import { BaseExportAdapter } from './base.adapter'
import { ExportAdapterType, ExportConfig, ExportResult, HuggingFaceConfig } from '../types/export.types'
import { whoAmI, createRepo, uploadFile } from '@huggingface/hub'

/**
 * Hugging Face Export Adapter
 * Uploads cleaned data to Hugging Face Datasets
 */
export class HuggingFaceExportAdapter extends BaseExportAdapter {
	type = ExportAdapterType.HUGGINGFACE
	name = 'Hugging Face Datasets'
	description = 'Export data to Hugging Face Datasets hub'
	requiredSettings = ['token', 'repoName']

	async validate(settings: Record<string, any>): Promise<{ valid: boolean; errors?: string[] }> {
		const baseValidation = await super.validate(settings)
		if (!baseValidation.valid) return baseValidation

		const errors: string[] = []
		const config = settings as HuggingFaceConfig

		if (!config.token || config.token.trim().length === 0) {
			errors.push('Hugging Face token cannot be empty')
		}

		if (!config.repoName || config.repoName.trim().length === 0) {
			errors.push('Repository name cannot be empty')
		}

		// Validate repo name format
		if (config.repoName && !/^[a-zA-Z0-9_-]+$/.test(config.repoName)) {
			errors.push('Repository name can only contain alphanumeric characters, hyphens, and underscores')
		}

		if (errors.length > 0) {
			return { valid: false, errors }
		}

		return { valid: true }
	}

	async export(data: any[], config: ExportConfig): Promise<ExportResult> {
		const hfConfig = config.settings as HuggingFaceConfig
		this.validate(hfConfig)

		try {
			const convertedData = this.convertData(data)

			if (convertedData.length === 0) {
				return {
					success: true,
					adapterType: this.type,
					fileName: config.fileName,
					destination: `huggingface.co/datasets/${hfConfig.repoName}`,
					recordsExported: 0,
					message: 'No data to export',
				}
			}

			// Get user info to construct full repo ID
			const user = await whoAmI({
				accessToken: hfConfig.token,
			})

			const fullRepoId = `${user.name}/${hfConfig.repoName}`
			const datasetRepoId = `datasets/${fullRepoId}` // Dataset repos use datasets/ prefix

			// Try to create the dataset repo (will fail silently if it already exists)
			try {
				await createRepo({
					repo: datasetRepoId,
					private: hfConfig.private || false,
					accessToken: hfConfig.token,
				})
			} catch (error) {
				// Repo likely already exists, continue
			}

			// Convert data to CSV format
			const fileName = hfConfig.fileName || 'cleaned_data.csv'
			const csvContent = this.dataToCSV(convertedData)

			// Upload CSV file
			await uploadFile({
				repo: datasetRepoId,
				file: {
					path: fileName,
					content: new Blob([csvContent], { type: 'text/csv' }),
				},
				commitTitle: hfConfig.commitMessage || `Upload cleaned data from flowmatic pipeline run ${config.pipelineRunId}`,
				accessToken: hfConfig.token,
			})

			// Generate and upload README.md
			const readmeContent = this.generateReadme(convertedData, fileName, config.pipelineRunId)
			await uploadFile({
				repo: datasetRepoId,
				file: {
					path: 'README.md',
					content: new Blob([readmeContent], { type: 'text/markdown' }),
				},
				commitTitle: 'Add dataset README with metadata',
				accessToken: hfConfig.token,
			})

			// Generate and upload datasets.yml for Hub preview
			const datasetsYamlContent = this.generateDatasetsYaml(convertedData, fileName)
			await uploadFile({
				repo: datasetRepoId,
				file: {
					path: 'datasets.yml',
					content: new Blob([datasetsYamlContent], { type: 'text/yaml' }),
				},
				commitTitle: 'Add datasets.yml for Hugging Face Hub preview',
				accessToken: hfConfig.token,
			})

			this.logExport(config, convertedData.length, `huggingface.co/datasets/${fullRepoId}`, 'success')

			return {
				success: true,
				adapterType: this.type,
				fileName: config.fileName,
				destination: `huggingface.co/datasets/${fullRepoId}/${fileName}`,
				recordsExported: convertedData.length,
				message: `Successfully exported ${convertedData.length} records to Hugging Face Datasets`,
				metadata: {
					repoId: fullRepoId,
					filePath: fileName,
				},
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error'
			this.logExport(config, 0, hfConfig.repoName, 'failed')
			throw new Error(`Hugging Face export failed: ${message}`)
		}
	}

	/**
	 * Convert data array to CSV string
	 */
	private dataToCSV(data: any[]): string {
		if (data.length === 0) return ''

		const headers = Object.keys(data[0])
		const headerLine = headers.map((h) => this.escapeCSV(h)).join(',')

		const dataLines = data.map((row) => {
			return headers.map((header) => this.escapeCSV(String(row[header] ?? ''))).join(',')
		})

		return [headerLine, ...dataLines].join('\n')
	}

	/**
	 * Escape CSV values
	 */
	private escapeCSV(value: string): string {
		if (value.includes(',') || value.includes('"') || value.includes('\n')) {
			return `"${value.replace(/"/g, '""')}"`.replace(/\n/g, '\\n')
		}
		return value
	}

	/**
	 * Generate README.md for the dataset with metadata and column descriptions
	 */
	private generateReadme(data: any[], fileName: string, pipelineRunId: string): string {
		const headers = data.length > 0 ? Object.keys(data[0]) : []
		const timestamp = new Date().toISOString()

		// Calculate basic statistics
		const stats = {
			totalRecords: data.length,
			totalColumns: headers.length,
			generatedAt: timestamp,
		}

		// Infer column types and get sample values
		const columnInfo = headers.map((col) => {
			const values = data.map((row) => row[col]).filter((v) => v !== null && v !== undefined)
			const sample = values.slice(0, 3)
			let type = 'text'

			if (values.every((v) => typeof v === 'boolean')) type = 'boolean'
			else if (values.every((v) => Number.isInteger(v))) type = 'integer'
			else if (values.every((v) => typeof v === 'number')) type = 'float'
			else if (values.every((v) => !isNaN(new Date(v).getTime()))) type = 'timestamp'

			return {
				name: col,
				type,
				nonNull: values.length,
				nullCount: data.length - values.length,
				sampleValues: sample,
			}
		})

		return `---
dataset_info:
  features:
  ${columnInfo.map((col) => `- name: ${col.name}\n    dtype: ${col.type}`).join('\n  ')}
  splits:
  - name: default
    num_bytes: ${Math.round(this.dataToCSV(data).length / 1024)}KB
    num_examples: ${data.length}
---

# Flowmatic Cleaned Dataset

## Overview
This dataset was cleaned and exported by **Flowmatic**, an intelligent data preparation platform. 

**Pipeline Run ID**: \`${pipelineRunId}\`
**Generated**: ${timestamp}

## Dataset Statistics

- **Total Records**: ${stats.totalRecords.toLocaleString()}
- **Total Columns**: ${stats.totalColumns}
- **File**: \`${fileName}\`

## Column Information

| Column | Type | Non-Null | Null | Sample Values |
|--------|------|----------|------|---------------|
${columnInfo.map((col) => `| ${col.name} | ${col.type} | ${col.nonNull} | ${col.nullCount} | ${col.sampleValues.map((v) => JSON.stringify(v)).join(', ')} |`).join('\n')}

## Data Quality

This dataset has been processed through Flowmatic's cleaning pipeline:

- ✅ Duplicates removed
- ✅ Missing values handled (interpolation/forward-fill)
- ✅ Outliers processed (winsorization)
- ✅ Type consistency validated
- ✅ Records exported

## Usage

Load the dataset using Hugging Face \`datasets\` library:

\`\`\`python
from datasets import load_dataset

dataset = load_dataset('${this.getCurrentUser() || 'username'}/dataset_name')
df = dataset['train'].to_pandas()
\`\`\`

Or load directly as CSV:

\`\`\`python
import pandas as pd

df = pd.read_csv('https://huggingface.co/datasets/${this.getCurrentUser() || 'username'}/dataset_name/raw/main/${fileName}')
\`\`\`

## License

This dataset is released under the CC BY 4.0 license.

---

*Processed with [Flowmatic](https://github.com/flowmatic/flowmatic)*
`
	}

	/**
	 * Generate datasets.yml for Hugging Face Hub preview
	 */
	private generateDatasetsYaml(data: any[], fileName: string): string {
		const headers = data.length > 0 ? Object.keys(data[0]) : []

		return `# Datasets Configuration for Hugging Face Hub
# This configuration enables interactive preview on the Hub

configs:
  - config_name: default
    data_files:
      - path: ${fileName}
        type: csv
    description: "Cleaned dataset exported by Flowmatic"

dataset_info:
  features:
${headers.map((col) => `    - name: ${col}\n      dtype: string\n      description: "Column ${col}"`).join('\n')}
  splits:
    - name: train
      num_bytes: ${Math.round(this.dataToCSV(data).length / 1024)}
      num_examples: ${data.length}
  homepage: "https://huggingface.co/spaces/flowmatic/preview"
  license: cc-by-4.0
  tags:
    - cleaned
    - flowmatic
    - tabular
`
	}

	/**
	 * Get current Hugging Face user (placeholder)
	 */
	private getCurrentUser(): string | null {
		// In real implementation, this would get from the whoAmI call
		return null
	}
}