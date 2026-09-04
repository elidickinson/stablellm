export function normalizeBaseUrl(value: string): string {
	const url = new URL(value.trim());
	if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("StableLLM URL must use HTTP or HTTPS");
	if (url.search || url.hash) throw new Error("StableLLM URL must not contain a query string or fragment");
	const path = url.pathname.replace(/\/+$/, "");
	url.pathname = path === "" || path.endsWith("/v1") ? path || "/v1" : `${path}/v1`;
	return url.toString().replace(/\/$/, "");
}
