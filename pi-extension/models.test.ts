import assert from "node:assert/strict";
import test from "node:test";
import { mapStableLlmModels } from "./models.ts";

test("maps published StableLLM metadata to Pi models", () => {
	const [model, race] = mapStableLlmModels([{
		id: "example",
		name: "Example",
		default_mode: "seq",
		context_length: 100_000,
		architecture: { input_modalities: ["text", "image", "audio"] },
		pricing: {
			prompt: "0.000001",
			completion: "0.000003",
			input_cache_read: "0.0000001",
			input_cache_write: "0.000003",
		},
		top_provider: { max_completion_tokens: 8_000 },
		reasoning: { mandatory: false, supported_efforts: ["none", "low", "high"] },
	}]);

	assert.deepEqual(model, {
		id: "example",
		name: "Example",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 1, output: 3, cacheRead: 0.1, cacheWrite: 3 },
		contextWindow: 100_000,
		maxTokens: 8_000,
		thinkingLevelMap: {
			off: "none",
			minimal: null,
			low: "low",
			medium: null,
			high: "high",
			xhigh: null,
			max: null,
		},
		compat: { supportsDeveloperRole: false, supportsReasoningEffort: true },
	});
	assert.equal(race.id, "example:race");
	assert.equal(race.name, "Example (race)");
});

test("does not add a sequential alias for race-default groups", () => {
	const models = mapStableLlmModels([{ id: "fast", default_mode: "race" }]);
	assert.deepEqual(models.map(({ id }) => id), ["fast"]);
});

test("uses known defaults when a familiar group omits metadata", () => {
	const [model] = mapStableLlmModels([{ id: "glm-5.2" }]);
	assert.equal(model.name, "GLM 5.2");
	assert.equal(model.contextWindow, 250_000);
	assert.equal(model.maxTokens, 131_072);
	assert.equal(model.reasoning, true);
	assert.deepEqual(model.thinkingLevelMap, {
		high: "high",
		max: "max",
		minimal: null,
		low: null,
		medium: null,
		xhigh: null,
	});
});

test("published metadata overrides known defaults", () => {
	const [model] = mapStableLlmModels([{ id: "glm-5.2", context_length: 64_000 }]);
	assert.equal(model.contextWindow, 64_000);
});

test("uses conservative Pi defaults when metadata is absent", () => {
	const [model, race] = mapStableLlmModels([{ id: "plain" }]);
	assert.deepEqual(model, {
		id: "plain",
		name: "plain",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 16_384,
		compat: { supportsDeveloperRole: false },
	});
	assert.equal(race.id, "plain:race");
});

test("mandatory reasoning hides off when efforts are unspecified", () => {
	const [model] = mapStableLlmModels([{ id: "reasoner", reasoning: { mandatory: true } }]);
	assert.deepEqual(model.thinkingLevelMap, { off: null });
});

test("applies the DeepSeek replay compatibility requirement", () => {
	const [model] = mapStableLlmModels([{ id: "deepseek-v4-flash", reasoning: {} }]);
	assert.equal(model.compat.requiresReasoningContentOnAssistantMessages, true);
});
