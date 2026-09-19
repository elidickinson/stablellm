import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import type { AddressInfo } from "node:net";
import { createRaceFetch, pendingRaceTarget, registerRaceFeedback, RACE_WORKING_MESSAGE, withRaceFetch } from "./race.ts";

interface UpstreamRequest {
	method?: string;
	url?: string;
	body: string;
	authorization?: string | string[];
}

interface UpstreamReply {
	status: number;
	body: string;
	headers?: Record<string, string>;
}

async function withServer(
	handler: (request: UpstreamRequest) => UpstreamReply,
	run: (baseUrl: string) => Promise<void>,
): Promise<void> {
	const server = createServer((req, res) => {
		let raw = "";
		req.on("data", (chunk) => (raw += chunk));
		req.on("end", () => {
			const reply = handler({ method: req.method, url: req.url, body: raw, authorization: req.headers.authorization });
			res.writeHead(reply.status, { "content-type": "application/json", ...reply.headers });
			res.end(reply.body);
		});
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const { port } = server.address() as AddressInfo;
	try {
		await run(`http://127.0.0.1:${port}`);
	} finally {
		await new Promise<void>((resolve) => server.close(() => resolve()));
	}
}

const TEST_ORIGIN = "https://stablellm.test";

function redirect(status: number, location: string, headers: Record<string, string> = {}): Response {
	return new Response("", { status, headers: { location, ...headers } });
}

test("pendingRaceTarget accepts only the same-origin marked 307 handshake", () => {
	const marked = { "x-stablellm-race": "pending" };
	assert.equal(
		pendingRaceTarget(redirect(307, "/v1/chat/completions?a=1&stablellm_race_redirect=1", marked), `${TEST_ORIGIN}/v1/chat/completions`),
		`${TEST_ORIGIN}/v1/chat/completions?a=1&stablellm_race_redirect=1`,
	);
	assert.equal(pendingRaceTarget(redirect(307, "https://evil.test/v1/chat/completions?stablellm_race_redirect=1", marked), `${TEST_ORIGIN}/v1/chat/completions`), undefined, "cross-origin is refused");
	assert.equal(pendingRaceTarget(redirect(307, "/v1/chat/completions", marked), `${TEST_ORIGIN}/v1/chat/completions`), undefined, "missing marker is refused");
	assert.equal(pendingRaceTarget(redirect(302, "/v1/chat/completions?stablellm_race_redirect=1", marked), `${TEST_ORIGIN}/v1/chat/completions`), undefined, "non-307 is refused");
	assert.equal(
		pendingRaceTarget(redirect(307, "/v1/chat/completions?stablellm_race_redirect=1", { "x-stablellm-race": "nope" }), `${TEST_ORIGIN}/v1/chat/completions`),
		undefined,
		"missing pending header is refused",
	);
});

test("follows the race handshake, replaying method, body, headers, and signal", async () => {
	const seen: UpstreamRequest[] = [];
	let notified = 0;
	await withServer(
		(request) => {
			seen.push(request);
			if (request.url === "/v1/chat/completions") {
				return { status: 307, body: "", headers: { location: "/v1/chat/completions?stablellm_race_redirect=1", "x-stablellm-race": "pending" } };
			}
			return { status: 200, body: JSON.stringify({ served: "final" }) };
		},
		async (baseUrl) => {
			const fetchImpl = createRaceFetch({ onRacePending: () => (notified += 1) });
			const controller = new AbortController();
			const response = await fetchImpl(`${baseUrl}/v1/chat/completions`, {
				method: "POST",
				headers: { authorization: "Bearer secret", "content-type": "application/json" },
				body: JSON.stringify({ model: "m", stream: true }),
				signal: controller.signal,
			});
			assert.equal(response.status, 200);
			assert.deepEqual(await response.json(), { served: "final" });
		},
	);
	assert.equal(notified, 1);
	assert.equal(seen.length, 2);
	assert.equal(seen[0].method, "POST");
	assert.equal(seen[1].method, "POST");
	assert.equal(seen[1].body, seen[0].body);
	assert.equal(seen[1].authorization, "Bearer secret");
	assert.equal(seen[1].url, "/v1/chat/completions?stablellm_race_redirect=1");
});

test("passes ordinary responses through untouched and follows a stranger redirect normally", async () => {
	let notified = 0;
	await withServer(
		() => ({ status: 200, body: JSON.stringify({ ok: true }) }),
		async (baseUrl) => {
			const fetchImpl = createRaceFetch({ onRacePending: () => (notified += 1) });
			const response = await fetchImpl(`${baseUrl}/v1/chat/completions`, { method: "POST", body: "{}" });
			assert.equal(response.status, 200);
			assert.deepEqual(await response.json(), { ok: true });
		},
	);
	assert.equal(notified, 0);

	const followed: string[] = [];
	await withServer(
		(request) => {
			followed.push(request.url ?? "");
			return request.url === "/v1/chat/completions"
				? { status: 307, body: "", headers: { location: "/elsewhere" } }
				: { status: 200, body: JSON.stringify({ where: "elsewhere" }) };
		},
		async (baseUrl) => {
			const fetchImpl = createRaceFetch({ onRacePending: () => (notified += 1) });
			const response = await fetchImpl(`${baseUrl}/v1/chat/completions`);
			assert.deepEqual(await response.json(), { where: "elsewhere" });
		},
	);
	assert.deepEqual(followed, ["/v1/chat/completions", "/elsewhere"]);
	assert.equal(notified, 0);
});

test("ordinary cross-origin redirects strip credentials and preserve fetch method semantics", async () => {
	let destination: UpstreamRequest | undefined;
	await withServer(
		(request) => {
			destination = request;
			return { status: 200, body: JSON.stringify({ ok: true }) };
		},
		async (destinationUrl) => {
			await withServer(
				() => ({ status: 302, body: "", headers: { location: `${destinationUrl}/moved` } }),
				async (sourceUrl) => {
					const fetchImpl = createRaceFetch({ onRacePending: () => {} });
					const response = await fetchImpl(`${sourceUrl}/v1/chat/completions`, {
						method: "POST",
						headers: { authorization: "Bearer secret", "content-type": "application/json" },
						body: "{}",
					});
					assert.equal(response.status, 200);
				},
			);
		},
	);
	assert.equal(destination?.method, "GET");
	assert.equal(destination?.body, "");
	assert.equal(destination?.authorization, undefined);
});

test("rejects a repeated pending redirect instead of looping", async () => {
	let requests = 0;
	await withServer(
		() => {
			requests += 1;
			return { status: 307, body: "", headers: { location: "/v1/chat/completions?stablellm_race_redirect=1", "x-stablellm-race": "pending" } };
		},
		async (baseUrl) => {
			const fetchImpl = createRaceFetch({ onRacePending: () => {} });
			await assert.rejects(fetchImpl(`${baseUrl}/v1/chat/completions`, { method: "POST", body: "{}" }), /refusing to follow it again/);
		},
	);
	assert.equal(requests, 2);
});

test("injects the race fetch into both stream entry points", () => {
	const calls: string[] = [];
	const streams = {
		stream: (_model: unknown, _context: unknown, options?: { fetch?: unknown }) => {
			calls.push(`stream:${options?.fetch ? "fetch" : "none"}`);
			return undefined as never;
		},
		streamSimple: (_model: unknown, _context: unknown, options?: { fetch?: unknown }) => {
			calls.push(`streamSimple:${options?.fetch ? "fetch" : "none"}`);
			return undefined as never;
		},
	};
	const wrapped = withRaceFetch(streams as never, () => {});
	wrapped.stream({} as never, {} as never);
	wrapped.streamSimple({} as never, {} as never);
	assert.deepEqual(calls, ["stream:fetch", "streamSimple:fetch"]);
	assert.equal(RACE_WORKING_MESSAGE, "StableLLM is racing providers...");
});

test("working message appears on race and clears on every turn lifecycle exit", () => {
	const handlers = new Map<string, (event: unknown, ctx: unknown) => unknown>();
	const pi = {
		on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => handlers.set(event, handler),
	} as never;
	const feedback = registerRaceFeedback(pi);

	const working: Array<string | undefined> = [];
	const ctx = { ui: { setWorkingMessage: (message?: string) => working.push(message) } };
	const last = () => working.at(-1);
	const assistant = { message: { role: "assistant" } };

	handlers.get("turn_start")!({}, ctx);
	assert.equal(last(), undefined);
	feedback.onRacePending();
	assert.equal(last(), RACE_WORKING_MESSAGE);

	for (const event of ["after_provider_response", "message_end", "turn_end", "model_select", "session_shutdown"]) {
		feedback.onRacePending();
		handlers.get(event)!(assistant, ctx);
		assert.equal(last(), undefined, `${event} clears`);
	}

	const count = working.length;
	handlers.get("message_end")!({ message: { role: "user" } }, ctx);
	assert.equal(working.length, count);
});
