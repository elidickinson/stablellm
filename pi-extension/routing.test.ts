import assert from "node:assert/strict";
import test from "node:test";
import { parseStableLlmHeaders } from "./routing.ts";

test("extracts the resolved StableLLM upstream", () => {
	assert.deepEqual(parseStableLlmHeaders({
		"x-stablellm-provider": "cerebras",
		"x-stablellm-model": "zai-glm-5.3",
	}), { provider: "cerebras", model: "zai-glm-5.3" });
});

test("ignores unrelated responses", () => {
	assert.equal(parseStableLlmHeaders({ "content-type": "text/event-stream" }), undefined);
});
