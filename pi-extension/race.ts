import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { FetchFunction, ProviderStreams } from "@earendil-works/pi-ai/compat";

export const RACE_PENDING_HEADER = "x-stablellm-race";
export const RACE_PENDING_VALUE = "pending";
export const RACE_MARKER_PARAM = "stablellm_race_redirect";
export const RACE_MARKER_VALUE = "1";
export const RACE_WORKING_MESSAGE = "StableLLM is racing providers...";

const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

function isRedirect(response: Response): boolean {
	return REDIRECT_STATUSES.has(response.status);
}

/**
 * Resolves a redirect target when, and only when, the response is the
 * server-confirmed race handshake: a 307 bearing `X-StableLLM-Race: pending`
 * whose Location stays on the request's origin and carries the race marker.
 * Anything else (cross-origin, no marker, ordinary redirect) is left alone.
 */
export function pendingRaceTarget(response: Response, requestUrl: string): string | undefined {
	if (response.status !== 307) return undefined;
	if (response.headers.get(RACE_PENDING_HEADER) !== RACE_PENDING_VALUE) return undefined;
	const location = response.headers.get("location");
	if (!location) return undefined;
	const origin = new URL(requestUrl);
	const target = new URL(location, origin);
	if (target.origin !== origin.origin) return undefined;
	if (target.searchParams.get(RACE_MARKER_PARAM) !== RACE_MARKER_VALUE) return undefined;
	return target.toString();
}

export interface RaceFetchOptions {
	/** Called once the server has confirmed a race is about to run. */
	onRacePending: () => void;
	/** Base fetch implementation; defaults to the global fetch. */
	fetchImpl?: FetchFunction;
}

/**
 * Fetch that follows the race handshake for the OpenAI SDK: the SDK treats the
 * server's 307 as an API error, so the wrapper replays the POST to the marked
 * Location itself and hands the SDK only the final response. Ordinary
 * responses and redirects keep normal fetch behavior, the follow-up is never
 * followed manually again (a repeated pending redirect throws instead of
 * looping), and auth never leaves the request's origin.
 */
export function createRaceFetch({ onRacePending, fetchImpl = globalThis.fetch }: RaceFetchOptions): FetchFunction {
	return async (input, init) => {
		const request = new Request(input, init);
		// Buffer the body so the marked follow-up can replay the exact POST.
		const body = await request.arrayBuffer();
		const originalBody = body.byteLength > 0 ? body : undefined;
		const send = (
			url: string,
			redirect: RequestRedirect,
			method = request.method,
			headers: HeadersInit = request.headers,
			requestBody: BodyInit | null = originalBody ?? null,
		) => fetchImpl(url, { method, headers, body: requestBody ?? undefined, signal: request.signal, redirect });

		const followLocation = async (response: Response, base: string) => {
			const location = response.headers.get("location");
			if (!location) return response;

			const target = new URL(location, base);
			const headers = new Headers(request.headers);
			if (target.origin !== new URL(base).origin) {
				for (const name of ["authorization", "proxy-authorization", "cookie", "cookie2"]) headers.delete(name);
			}

			let method = request.method;
			let redirectBody: BodyInit | null = originalBody ?? null;
			const switchesToGet =
				(response.status === 303 && method !== "GET" && method !== "HEAD") ||
				((response.status === 301 || response.status === 302) && method === "POST");
			if (switchesToGet) {
				method = "GET";
				redirectBody = null;
				for (const name of ["content-encoding", "content-language", "content-location", "content-type", "content-length"]) headers.delete(name);
			}
			await response.body?.cancel();
			return send(target.toString(), "follow", method, headers, redirectBody);
		};

		const first = await send(request.url, "manual");
		const target = pendingRaceTarget(first, request.url);
		if (!target) return isRedirect(first) ? followLocation(first, request.url) : first;

		await first.body?.cancel();
		onRacePending();
		const follow = await send(target, "manual");
		if (pendingRaceTarget(follow, target)) {
			await follow.body?.cancel();
			throw new Error("StableLLM race redirect repeated; refusing to follow it again");
		}
		return isRedirect(follow) ? followLocation(follow, target) : follow;
	};
}

/** Wraps every provider stream entry so its request carries the race fetch. */
export function withRaceFetch(streams: ProviderStreams, onRacePending: () => void): ProviderStreams {
	return {
		...streams,
		stream: (model, context, options) =>
			streams.stream(model, context, {
				...options,
				fetch: createRaceFetch({ onRacePending, fetchImpl: options?.fetch }),
			}),
		streamSimple: (model, context, options) =>
			streams.streamSimple(model, context, {
				...options,
				fetch: createRaceFetch({ onRacePending, fetchImpl: options?.fetch }),
			}),
	};
}

/**
 * Out-of-band wiring between the fetch wrapper and Pi's UI: the wrapper has no
 * extension context of its own, so the latest one seen in turn events is kept
 * here. The working message is set when a race is confirmed and cleared on the
 * final response and every way a turn can end.
 */
export function registerRaceFeedback(pi: ExtensionAPI): { onRacePending: () => void } {
	let context: ExtensionContext | undefined;

	const clear = (ctx: ExtensionContext | undefined) => {
		(ctx ?? context)?.ui.setWorkingMessage();
	};

	pi.on("turn_start", (_event, ctx) => {
		context = ctx;
		clear(ctx);
	});
	pi.on("after_provider_response", (_event, ctx) => clear(ctx));
	pi.on("message_end", (event, ctx) => {
		if (event.message.role === "assistant") clear(ctx);
	});
	pi.on("turn_end", (_event, ctx) => clear(ctx));
	pi.on("model_select", (_event, ctx) => clear(ctx));
	pi.on("session_shutdown", (_event, ctx) => clear(ctx));

	return {
		onRacePending: () => context?.ui.setWorkingMessage(RACE_WORKING_MESSAGE),
	};
}
