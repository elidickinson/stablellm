export type StableLlmMode = "seq" | "race";

type StableLlmPricing = {
	prompt?: string;
	completion?: string;
	input_cache_read?: string;
	input_cache_write?: string;
};

type StableLlmReasoning = {
	mandatory?: boolean;
	default_enabled?: boolean;
	supported_efforts?: string[];
};

export interface StableLlmCatalogModel {
	id: string;
	name?: string;
	default_mode?: StableLlmMode;
	context_length?: number;
	architecture?: { input_modalities?: string[] };
	pricing?: StableLlmPricing;
	top_provider?: {
		context_length?: number;
		max_completion_tokens?: number;
	};
	reasoning?: StableLlmReasoning;
}

export interface PiModelDefinition {
	id: string;
	name: string;
	reasoning: boolean;
	input: Array<"text" | "image">;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
	maxTokens: number;
	thinkingLevelMap?: Partial<Record<"off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max", string | null>>;
	compat: {
		supportsDeveloperRole: false;
		supportsReasoningEffort?: true;
		requiresReasoningContentOnAssistantMessages?: true;
	};
}

const PI_REASONING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;
const REASONING_CONTENT_MODELS = new Set([
	"deepseek-v4-flash",
	"deepseek-v4-flash-fast",
	"deepseek-v4-flash-exacto",
]);

const DEEPSEEK_DEFAULTS: Partial<StableLlmCatalogModel> = {
	context_length: 1_000_000,
	top_provider: { max_completion_tokens: 65_536 },
	architecture: { input_modalities: ["text"] },
	pricing: {
		prompt: "0.00000014",
		completion: "0.00000028",
		input_cache_read: "0.000000028",
		input_cache_write: "0",
	},
	reasoning: { supported_efforts: ["none", "high", "max"] },
};

const KNOWN_MODEL_DEFAULTS: Record<string, Partial<StableLlmCatalogModel>> = {
	"minimax-m2.5": {
		name: "MiniMax M2.5",
		context_length: 204_800,
		top_provider: { max_completion_tokens: 131_072 },
		architecture: { input_modalities: ["text"] },
		pricing: {
			prompt: "0.0000003",
			completion: "0.0000012",
			input_cache_read: "0.00000006",
			input_cache_write: "0.000000375",
		},
		reasoning: {},
	},
	"kimi-k3": {
		name: "Kimi K3",
		context_length: 524_288,
		top_provider: { max_completion_tokens: 65_536 },
		architecture: { input_modalities: ["text", "image"] },
		reasoning: { supported_efforts: ["max"] },
	},
	"glm-5.2": {
		name: "GLM 5.2",
		context_length: 250_000,
		top_provider: { max_completion_tokens: 131_072 },
		architecture: { input_modalities: ["text"] },
		reasoning: { supported_efforts: ["high", "max"] },
	},
	"glm-5.3": {
		name: "GLM 5.3",
		architecture: { input_modalities: ["text"] },
		reasoning: { mandatory: true, supported_efforts: ["low", "high", "max"] },
	},
	"glm-5.3-flash": {
		name: "GLM 5.3 Flash",
		architecture: { input_modalities: ["text"] },
		reasoning: { mandatory: true, supported_efforts: ["low", "high", "max"] },
	},
	"deepseek-v4-flash": DEEPSEEK_DEFAULTS,
	"deepseek-v4-flash-fast": { ...DEEPSEEK_DEFAULTS, name: "DeepSeek V4 Flash Fast" },
	"deepseek-v4-flash-exacto": { ...DEEPSEEK_DEFAULTS, name: "DeepSeek V4 Flash Exacto" },
	"qwen3.8-max": {
		name: "Qwen3.8 Max",
		context_length: 1_000_000,
		top_provider: { max_completion_tokens: 131_072 },
		architecture: { input_modalities: ["text"] },
		reasoning: { supported_efforts: ["low", "medium", "xhigh"] },
	},
	"qwen3.8-flash": {
		name: "Qwen3.8 Flash",
		architecture: { input_modalities: ["text"] },
		reasoning: {},
	},
	"qwen-3.8-27b": {
		name: "Qwen 3.8 27B",
		architecture: { input_modalities: ["text"] },
		reasoning: { supported_efforts: ["low", "medium", "xhigh"] },
	},
};

function withKnownDefaults(model: StableLlmCatalogModel): StableLlmCatalogModel {
	const defaults = KNOWN_MODEL_DEFAULTS[model.id];
	if (!defaults) return model;
	return {
		...defaults,
		...model,
		architecture: { ...defaults.architecture, ...model.architecture },
		pricing: { ...defaults.pricing, ...model.pricing },
		top_provider: { ...defaults.top_provider, ...model.top_provider },
		reasoning: model.reasoning ? { ...defaults.reasoning, ...model.reasoning } : defaults.reasoning,
	};
}

function perMillion(value: string | undefined): number {
	if (value === undefined) return 0;
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed >= 0 ? Number((parsed * 1_000_000).toPrecision(15)) : 0;
}

function positiveInt(...values: Array<number | undefined>): number | undefined {
	return values.find((value): value is number => typeof value === "number" && Number.isInteger(value) && value > 0);
}

function inputModalities(model: StableLlmCatalogModel): Array<"text" | "image"> {
	const modalities = model.architecture?.input_modalities ?? [];
	const input = modalities.filter((value): value is "text" | "image" => value === "text" || value === "image");
	return input.length > 0 ? [...new Set(input)] : ["text"];
}

function thinkingLevelMap(reasoning: StableLlmReasoning): PiModelDefinition["thinkingLevelMap"] {
	const efforts = new Set(reasoning.supported_efforts ?? []);
	if (efforts.size === 0) return reasoning.mandatory ? { off: null } : undefined;

	const levels: NonNullable<PiModelDefinition["thinkingLevelMap"]> = {};
	if (reasoning.mandatory || (reasoning.default_enabled && !efforts.has("none"))) levels.off = null;
	else if (efforts.has("none")) levels.off = "none";
	for (const level of PI_REASONING_LEVELS) levels[level] = efforts.has(level) ? level : null;
	return levels;
}

function mapOne(model: StableLlmCatalogModel, id = model.id, name = model.name ?? model.id): PiModelDefinition {
	const reasoning = model.reasoning !== undefined;
	return {
		id,
		name,
		reasoning,
		input: inputModalities(model),
		cost: {
			input: perMillion(model.pricing?.prompt),
			output: perMillion(model.pricing?.completion),
			cacheRead: perMillion(model.pricing?.input_cache_read),
			cacheWrite: perMillion(model.pricing?.input_cache_write),
		},
		contextWindow: positiveInt(model.context_length, model.top_provider?.context_length) ?? 128_000,
		maxTokens: positiveInt(model.top_provider?.max_completion_tokens) ?? 16_384,
		...(reasoning ? { thinkingLevelMap: thinkingLevelMap(model.reasoning!) } : {}),
		compat: {
			supportsDeveloperRole: false,
			...(reasoning ? { supportsReasoningEffort: true as const } : {}),
			...(REASONING_CONTENT_MODELS.has(model.id)
				? { requiresReasoningContentOnAssistantMessages: true as const }
				: {}),
		},
	};
}

export function mapStableLlmModels(models: StableLlmCatalogModel[]): PiModelDefinition[] {
	return models.flatMap((catalogModel) => {
		const model = withKnownDefaults(catalogModel);
		const base = mapOne(model);
		if (model.default_mode === "race") return [base];
		return [base, mapOne(model, `${model.id}:race`, `${model.name ?? model.id} (race)`)];
	});
}
