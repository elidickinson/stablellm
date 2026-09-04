import assert from "node:assert/strict";
import test from "node:test";
import { normalizeBaseUrl } from "./url.ts";

test("normalizes root and OpenAI-compatible StableLLM URLs", () => {
	assert.equal(normalizeBaseUrl("https://example.test"), "https://example.test/v1");
	assert.equal(normalizeBaseUrl("https://example.test/v1/"), "https://example.test/v1");
	assert.equal(normalizeBaseUrl("https://example.test/proxy"), "https://example.test/proxy/v1");
	// Already-normalized URLs pass through unchanged (idempotent).
	assert.equal(normalizeBaseUrl("https://example.test/proxy/v1"), "https://example.test/proxy/v1");
	assert.equal(normalizeBaseUrl("https://example.test/v1/v1"), "https://example.test/v1/v1");
});

test("rejects non-HTTP URLs", () => {
	assert.throws(() => normalizeBaseUrl("file:///tmp/stablellm"), /HTTP or HTTPS/);
});
