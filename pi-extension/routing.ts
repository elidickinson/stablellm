export interface RoutingMetadata {
	provider?: string;
	model?: string;
}

function headerValue(headers: Record<string, string>, name: string): string | undefined {
	const value = headers[name]?.trim();
	return value || undefined;
}

export function parseStableLlmHeaders(headers: Record<string, string> | null): RoutingMetadata | undefined {
	if (!headers) return undefined;
	const provider = headerValue(headers, "x-stablellm-provider");
	const model = headerValue(headers, "x-stablellm-model");
	return provider || model ? { provider, model } : undefined;
}
