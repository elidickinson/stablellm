import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { parseStableLlmHeaders } from "./routing.js";

const PROVIDER_ID = "stablellm";
const STATUS_KEY = "stablellm-upstream";

export function registerUpstreamStatus(pi: ExtensionAPI): void {
	let responseHeaders: Record<string, string> | null = null;

	pi.on("turn_start", () => {
		responseHeaders = null;
	});

	pi.on("after_provider_response", (event) => {
		responseHeaders = event.headers;
	});

	pi.on("message_end", (event, ctx) => {
		if (event.message.role !== "assistant" || event.message.provider !== PROVIDER_ID) return;
		const upstream = parseStableLlmHeaders(responseHeaders);
		responseHeaders = null;
		if (!upstream) return;
		const route = [upstream.provider, upstream.model].filter(Boolean).join("/");
		ctx.ui.setStatus(STATUS_KEY, ctx.ui.theme.fg("dim", `last ${route}`));
	});

	pi.on("model_select", (_event, ctx) => {
		ctx.ui.setStatus(STATUS_KEY, undefined);
	});

	pi.on("session_shutdown", (_event, ctx) => {
		ctx.ui.setStatus(STATUS_KEY, undefined);
	});
}
