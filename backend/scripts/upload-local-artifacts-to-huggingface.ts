import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { basename, join, relative, sep } from 'node:path'

import { createRepo, uploadFile, whoAmI } from '@huggingface/hub'

type LocalArtifactRoot = {
	localPath: string
	repoPath: string
}

const defaultRoots: LocalArtifactRoot[] = [
	{ localPath: '../models/datasets', repoPath: 'models/datasets' },
	{ localPath: '../models/checkpoints', repoPath: 'models/checkpoints' },
	{ localPath: '../models/artifacts', repoPath: 'models/artifacts' },
	{ localPath: './model-artifacts', repoPath: 'backend/model-artifacts' },
	{ localPath: '../data', repoPath: 'data' },
]

function loadDotEnvToken(): string | undefined {
	const envPaths = [join(process.cwd(), '..', '.env'), join(process.cwd(), '.env')]

	for (const envPath of envPaths) {
		if (!existsSync(envPath)) continue

		const content = readFileSync(envPath, 'utf8')
		for (const rawLine of content.split(/\r?\n/)) {
			const line = rawLine.trim()
			if (!line || line.startsWith('#')) continue

			const separatorIndex = line.indexOf('=')
			if (separatorIndex === -1) {
				if (line.startsWith('hf_')) return line
				continue
			}

			const key = line.slice(0, separatorIndex).trim()
			const value = line
				.slice(separatorIndex + 1)
				.trim()
				.replace(/^['"]|['"]$/g, '')

			if (['HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGING_FACE_TOKEN'].includes(key)) {
				return value
			}
		}
	}

	return undefined
}

function getArgValue(name: string): string | undefined {
	const prefix = `--${name}=`
	const match = process.argv.find((arg) => arg.startsWith(prefix))
	return match?.slice(prefix.length)
}

function listFiles(root: LocalArtifactRoot): Array<{ localPath: string; repoPath: string }> {
	if (!existsSync(root.localPath)) return []

	const files: Array<{ localPath: string; repoPath: string }> = []
	const pending = [root.localPath]

	while (pending.length > 0) {
		const current = pending.pop()
		if (!current) continue

		for (const entry of readdirSync(current, { withFileTypes: true })) {
			if (entry.name === '__pycache__' || entry.name === '.cache') continue

			const entryPath = join(current, entry.name)
			if (entry.isDirectory()) {
				pending.push(entryPath)
				continue
			}

			if (!entry.isFile()) continue

			const nestedPath = relative(root.localPath, entryPath).split(sep).join('/')
			files.push({
				localPath: entryPath,
				repoPath: `${root.repoPath}/${nestedPath}`,
			})
		}
	}

	return files.sort((left, right) => left.repoPath.localeCompare(right.repoPath))
}

async function main(): Promise<void> {
	const token =
		process.env.HF_TOKEN ??
		process.env.HUGGINGFACE_TOKEN ??
		process.env.HUGGING_FACE_TOKEN ??
		loadDotEnvToken()

	if (!token) {
		throw new Error(
			'Missing Hugging Face token. Set HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_TOKEN, or put the hf_ token in .env.',
		)
	}

	const privateRepo = process.argv.includes('--private')
	const repoName = getArgValue('repo') ?? 'flowmatic-local-artifacts'
	const commitTitle = getArgValue('commit') ?? 'Upload Flowmatic local model artifacts'
	const user = await whoAmI({ accessToken: token })
	const repoId = `datasets/${user.name}/${repoName}`
	const allFiles = defaultRoots.flatMap(listFiles)

	if (allFiles.length === 0) {
		console.log('No local artifact files found to upload.')
		return
	}

	try {
		await createRepo({
			repo: repoId,
			private: privateRepo,
			accessToken: token,
		})
	} catch {
		// The repo already exists or the token has write access but cannot create it again.
	}

	console.log(`Uploading ${allFiles.length} files to https://huggingface.co/${repoId}`)

	for (const file of allFiles) {
		const sizeMb = (statSync(file.localPath).size / 1024 / 1024).toFixed(1)
		console.log(`- ${file.repoPath} (${sizeMb} MB)`)

		await uploadFile({
			repo: repoId,
			file: {
				path: file.repoPath,
				content: new Blob([readFileSync(file.localPath)]),
			},
			commitTitle: `${commitTitle}: ${basename(file.localPath)}`,
			accessToken: token,
			useXet: true,
		})
	}

	console.log(`Done: https://huggingface.co/${repoId}`)
}

main().catch((error: unknown) => {
	const message = error instanceof Error ? error.message : String(error)
	console.error(message)
	process.exitCode = 1
})
