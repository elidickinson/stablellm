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

test("falls back to conservative defaults when metadata is absent", () => {
	const [model, race] = mapStableLlmModels([{ id: "plain" }]);
	assert.deepEqual(model, {
		id: "plain",
		name: "plain",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 256_000,
		maxTokens: 16_384,
		compat: { supportsDeveloperRole: false },
	});
	assert.equal(race.id, "plain:race");
});

test("published metadata overrides the defaults field by field", () => {
	const [model] = mapStableLlmModels([{ id: "partial", context_length: 64_000, pricing: { prompt: "0.0000005" } }]);
	assert.equal(model.contextWindow, 64_000);
	assert.equal(model.maxTokens, 16_384);
	assert.equal(model.cost.input, 0.5);
	assert.equal(model.cost.output, 0);
});

test("does not add a sequential alias for race-default groups", () => {
	const models = mapStableLlmModels([{ id: "fast", default_mode: "race" }]);
	assert.deepEqual(models.map(({ id }) => id), ["fast"]);
});

test("sends the server's default effort when a group declares reasoning without efforts", () => {
	const [model] = mapStableLlmModels([{ id: "unspecified", reasoning: { default_effort: "high" } }]);
	assert.deepEqual(model.thinkingLevelMap, {
		minimal: "high",
		low: "high",
		medium: "high",
		high: "high",
		xhigh: null,
		max: null,
	});
});

test("hides every level when a group declares neither efforts nor a default", () => {
	const [model] = mapStableLlmModels([{ id: "silent", reasoning: {} }]);
	assert.deepEqual(model.thinkingLevelMap, {
		minimal: null,
		low: null,
		medium: null,
		high: null,
		xhigh: null,
		max: null,
	});
});

test("mandatory reasoning hides off", () => {
	const [model] = mapStableLlmModels([{ id: "reasoner", reasoning: { mandatory: true, default_effort: "high" } }]);
	assert.deepEqual(model.thinkingLevelMap, {
		off: null,
		minimal: "high",
		low: "high",
		medium: "high",
		high: "high",
		xhigh: null,
		max: null,
	});
});

test("sets supportsReasoningEffort only for reasoning models", () => {
	const [withReasoning] = mapStableLlmModels([{ id: "r", reasoning: { default_effort: "high" } }]);
	const [withoutReasoning] = mapStableLlmModels([{ id: "p" }]);
	assert.equal(withReasoning.compat.supportsReasoningEffort, true);
	assert.equal("supportsReasoningEffort" in withoutReasoning.compat, false);
});
