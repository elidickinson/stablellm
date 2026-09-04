import {
	type ApiKeyCredential,
	type AuthContext,
	type AuthResult,
	createProvider,
	type Model,
	openAICompletionsApi,
	type RefreshModelsContext,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { mapStableLlmModels, type StableLlmCatalogModel } from "./models.js";
import { registerUpstreamStatus } from "./status.js";
import { normalizeBaseUrl } from "./url.js";

const PROVIDER_ID = "stablellm";
const BASE_URL_ENV = "STABLELLM_BASE_URL";
const API_KEY_ENV = "STABLELLM_API_KEY";

interface ModelsResponse {
	data: StableLlmCatalogModel[];
}

function credentialBaseUrl(credential: ApiKeyCredential | undefined): string | undefined {
	const value = credential?.env?.[BASE_URL_ENV];
	return typeof value === "string" && value.trim() ? normalizeBaseUrl(value) : undefined;
}

async function resolveBaseUrl(ctx: AuthContext, credential: ApiKeyCredential | undefined): Promise<string | undefined> {
	const configured = credentialBaseUrl(credential) ?? (await ctx.env(BASE_URL_ENV))?.trim();
	return configured ? normalizeBaseUrl(configured) : undefined;
}

async function fetchCatalog(baseUrl: string, apiKey: string | undefined, signal: AbortSignal): Promise<ModelsResponse> {
	const response = await fetch(`${baseUrl}/models`, {
		headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
		signal,
	});
	if (!response.ok) throw new Error(`StableLLM model discovery failed: HTTP ${response.status}`);
	const payload = (await response.json()) as Partial<ModelsResponse>;
	if (!Array.isArray(payload.data)) throw new Error("StableLLM /models response has no data array");
	if (payload.data.some((model) => typeof model?.id !== "string" || !model.id.trim())) {
		throw new Error("StableLLM /models response contains an invalid model id");
	}
	return { data: payload.data };
}

function toPiModels(catalog: StableLlmCatalogModel[], baseUrl: string): Model<"openai-completions">[] {
	return mapStableLlmModels(catalog).map((model) => ({
		...model,
		api: "openai-completions",
		provider: PROVIDER_ID,
		baseUrl,
	}));
}

export default function stableLlmExtension(pi: ExtensionAPI): void {
	const provider = createProvider({
		id: PROVIDER_ID,
		name: "StableLLM",
		auth: {
			apiKey: {
				name: "StableLLM server",
				login: async (interaction): Promise<ApiKeyCredential> => {
					const environmentUrl = process.env[BASE_URL_ENV]?.trim();
					const enteredUrl = await interaction.prompt({
						type: "text",
						message: "StableLLM server URL (/v1 is appended if missing)",
						placeholder: environmentUrl ?? "https://stablellm.example.com/v1",
					});
					const rawUrl = enteredUrl.trim() || environmentUrl;
					if (!rawUrl) throw new Error(`A server URL or ${BASE_URL_ENV} is required`);
					const baseUrl = normalizeBaseUrl(rawUrl);
					const enteredKey = (
						await interaction.prompt({ type: "secret", message: "StableLLM API key (optional)" })
					).trim();
					await fetchCatalog(baseUrl, enteredKey || process.env[API_KEY_ENV]?.trim(), interaction.signal);
					return {
						type: "api_key",
						key: enteredKey || undefined,
						env: { [BASE_URL_ENV]: baseUrl },
					};
				},
				check: async ({ ctx, credential }) => {
					const baseUrl = await resolveBaseUrl(ctx, credential);
					return baseUrl
						? { type: "api_key" as const, source: credential ? "stored credential" : BASE_URL_ENV }
						: undefined;
				},
				resolve: async ({ ctx, credential }): Promise<AuthResult | undefined> => {
					const baseUrl = await resolveBaseUrl(ctx, credential);
					if (!baseUrl) return undefined;
					// "local" satisfies pi's OpenAI client, which requires a key even when
					// the server has none configured; a keyless server ignores the header.
					const apiKey = credential?.key ?? (await ctx.env(API_KEY_ENV)) ?? "local";
					return {
						auth: { apiKey, baseUrl },
						env: { ...credential?.env, [BASE_URL_ENV]: baseUrl },
						source: credential ? "stored credential" : BASE_URL_ENV,
					};
				},
			},
		},
		models: [],
		fetchModels: async (context: RefreshModelsContext): Promise<readonly Model<"openai-completions">[]> => {
			const credential = context.credential?.type === "api_key" ? context.credential : undefined;
			const rawBaseUrl = credentialBaseUrl(credential) ?? process.env[BASE_URL_ENV]?.trim();
			if (!rawBaseUrl) return [];
			const baseUrl = normalizeBaseUrl(rawBaseUrl);
			const apiKey = credential?.key ?? process.env[API_KEY_ENV]?.trim();
			const catalog = await fetchCatalog(baseUrl, apiKey, context.signal);
			return toPiModels(catalog.data, baseUrl);
		},
		api: openAICompletionsApi(),
	});

	pi.registerProvider(provider);
	registerUpstreamStatus(pi);
}
