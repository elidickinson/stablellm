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
	default_effort?: string;
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
	};
}

const PI_REASONING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

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

// A group can declare reasoning without declaring which efforts it accepts, and
// then the effort vocabulary is unknown — pi's own level names must not be
// forwarded verbatim. Every level resolves to the effort the server names as its
// default, and the extended levels stay hidden because they would all mean the
// same thing. When the server names no default either, no level can be sent
// safely and every level is hidden.
function thinkingLevelMap(reasoning: StableLlmReasoning): NonNullable<PiModelDefinition["thinkingLevelMap"]> {
	const efforts = new Set(reasoning.supported_efforts ?? []);
	const unspecified = efforts.size === 0 ? reasoning.default_effort : undefined;
	const levels: NonNullable<PiModelDefinition["thinkingLevelMap"]> = {};
	if (reasoning.mandatory || (reasoning.default_enabled && !efforts.has("none"))) levels.off = null;
	else if (efforts.has("none")) levels.off = "none";
	for (const level of PI_REASONING_LEVELS) {
		if (unspecified !== undefined) levels[level] = level === "xhigh" || level === "max" ? null : unspecified;
		else levels[level] = efforts.has(level) ? level : null;
	}
	return levels;
}

// Pi model definitions carry a value for every field. Values the server doesn't
// publish fall back to defaults: a 256K context window, and Pi's own 16K maximum
// output. A cap above an upstream's real limit is a request error that costs the
// turn; one below it only truncates. Prices have no defensible guess, so they
// stay zero.
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
		contextWindow: positiveInt(model.context_length, model.top_provider?.context_length) ?? 256_000,
		maxTokens: positiveInt(model.top_provider?.max_completion_tokens) ?? 16_384,
		...(reasoning ? { thinkingLevelMap: thinkingLevelMap(model.reasoning!) } : {}),
		compat: {
			supportsDeveloperRole: false,
			...(reasoning ? { supportsReasoningEffort: true as const } : {}),
		},
	};
}

export function mapStableLlmModels(models: StableLlmCatalogModel[]): PiModelDefinition[] {
	return models.flatMap((model) => {
		const base = mapOne(model);
		if (model.default_mode === "race") return [base];
		return [base, mapOne(model, `${base.id}:race`, `${base.name} (race)`)];
	});
}
