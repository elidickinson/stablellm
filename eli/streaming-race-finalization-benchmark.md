# Benchmark task: streaming race does not publish its winner

## Base revision

- Repository: `stablellm`
- Checkout: `f16a6fcb05a16bd6a51a5a100923dde46a26f0be`
- Commit: `Make session pin TTL configurable as session_pin_ttl_secs, default 15m (was hardcoded 10m)`
- Captured: 2026-08-31

The bug exists at this revision. Do not use later commits when reproducing it.

## Task

Investigate and fix the streaming race-routing bug shown below. The first request to the `fast` group races its configured endpoints and reports `neuralwatt` as the winner. However, the race never publishes its completion/order result. The next request therefore starts sequential routing from the original configuration order and sends the request to `crofai` instead of trying the previous race winner first.

Keep the existing periodic-race behavior in scope: a group configured with `mode: race` may dispatch ordinary requests sequentially between races. The issue for this task is that the preferred order must reflect the most recently completed race. A request between races remains in the group's `race` mode; it uses the published preferred order without launching another race.

Add a focused regression test. Avoid changing unrelated routing, session-key, or concurrency behavior.

## Configuration

The relevant group is:

```yaml
settings:
  cooloff_seconds: 30
  race_interval_requests: 25
  log_level: DEBUG

fast:
  mode: race
  endpoints:
    - provider: crofai
      model: deepseek-v4-flash-0731
    - provider: openrouter
      model: openai/gpt-5-nano
    - provider: neuralwatt
      model: deepseek-v4-flash
    - provider: openrouter
      model: z-ai/glm-5.3-flash:nitro
```

The OpenRouter endpoint involved in the race is configured from this model entry:

```yaml
- provider: openrouter
  model: z-ai/glm-5.3-flash
```

Race candidates are grouped by endpoint model and base URL, not only by provider name. The two OpenRouter entries therefore produce two distinct candidates even though their log labels are both `openrouter`; the supplied configuration has four candidates total.

A successful candidate is accounted for only after its complete response body is drained. The winner's stream is accounted for when its streaming generator finishes, so it does not emit a separate `race: drain neuralwatt` line. An explicit candidate failure also counts as accounted. Consequently, the three loser drain lines plus the winner's terminal request line account for all four candidates in this capture. The slowest loser controls when `race complete` can be emitted; handling a never-ending loser is outside this benchmark task.

## Observed log

```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-08-31 20:34:58,900 INFO    stablellm: stablellm started with 37 endpoint(s), groups: ['glm-5.2', 'glm-5.3', 'glm-5.3-flash', 'deepseek-v4-flash', 'deepseek-v4-flash-fast', 'deepseek-v4-flash-exacto', 'qwen3.8-max', 'qwen-3.8-27b', 'kimi-k3', 'minimax-m2.5', 'fast', 'default', 'cwg', 'kidsweather']
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
2026-08-31 20:40:10,763 DEBUG   stablellm: req=06f03672e73bc917 -> model='fast' group=fast mode=race stream=True keyname=legacy
2026-08-31 20:40:10,764 DEBUG   stablellm: req=06f03672e73bc917 race: trigger=first request candidates=['crofai model=deepseek-v4-flash-0731', 'openrouter model=openai/gpt-5-nano', 'neuralwatt model=deepseek-v4-flash', 'openrouter model=z-ai/glm-5.3-flash:nitro'] gen=1
2026-08-31 20:40:11,513 DEBUG   stablellm: req=06f03672e73bc917 race: winner neuralwatt (model=deepseek-v4-flash) 749ms
2026-08-31 20:40:11,737 INFO    stablellm: req=06f03672e73bc917 200 model=fast served=deepseek-v4-flash provider=neuralwatt mode=race stream=yes ttfb=749ms ttft=750ms tokens=8 tok/s=36 keyname=legacy
2026-08-31 20:40:14,097 DEBUG   stablellm: req=06f03672e73bc917 race: drain crofai done 3.3s
2026-08-31 20:40:19,047 DEBUG   stablellm: req=06f03672e73bc917 race: drain openrouter done 8.3s
2026-08-31 20:40:47,134 DEBUG   stablellm: req=06f03672e73bc917 race: drain openrouter done 36.4s
2026-08-31 20:42:19,675 DEBUG   stablellm: req=a640ef55c0a8cac4 -> model='fast' group=fast mode=race stream=True keyname=legacy
2026-08-31 20:42:19,676 DEBUG   stablellm: req=a640ef55c0a8cac4 attempt 1/4 -> crofai model=deepseek-v4-flash-0731 body_keys=['model', 'messages', 'stream', 'stream_options', 'max_completion_tokens', 'temperature'] bytes=2227
2026-08-31 20:42:22,011 DEBUG   stablellm: req=a640ef55c0a8cac4 crofai TTFT 2335ms (TTFB 2335ms)
2026-08-31 20:42:37,804 INFO    stablellm: req=a640ef55c0a8cac4 200 model=fast served=deepseek-v4-flash-0731 provider=crofai mode=seq stream=yes ttfb=2335ms ttft=2335ms tokens=1158 tok/s=73 keyname=legacy
2026-08-31 20:43:02,619 DEBUG   stablellm: req=e58442869bd51d6b -> model='fast' group=fast mode=race stream=True keyname=legacy
2026-08-31 20:43:02,619 DEBUG   stablellm: req=e58442869bd51d6b pinned session -> crofai
2026-08-31 20:43:02,619 DEBUG   stablellm: req=e58442869bd51d6b attempt 1/4 -> crofai model=deepseek-v4-flash-0731 body_keys=['model', 'messages', 'stream', 'stream_options', 'max_completion_tokens', 'temperature'] bytes=2227
2026-08-31 20:43:03,873 DEBUG   stablellm: req=e58442869bd51d6b crofai TTFT 1253ms (TTFB 1253ms)
2026-08-31 20:43:15,555 INFO    stablellm: req=e58442869bd51d6b 200 model=fast served=deepseek-v4-flash-0731 provider=crofai mode=seq stream=yes ttfb=1253ms ttft=1253ms tokens=991 tok/s=85 keyname=legacy
```

## Expected behavior

- The first request races one endpoint per provider-group candidate.
- `neuralwatt` wins the race in this capture.
- After all racer responses have been accounted for, the implementation records a completion event and updates `fast`'s preferred provider order with the race results.
- Because the next request is inside the race interval, it uses the group's published preferred order without launching another race. Its mode remains `race`, and it should try `neuralwatt` first (unless it is unavailable or capped).
- A later request with the same derived conversation key may be pinned to the endpoint that served the immediately preceding sequential request. Racing itself must not create a session pin.

## Acceptance criteria

1. A streaming race with at least three candidates updates the group's preferred order after the winner stream and loser drains finish.
2. The completion log is emitted exactly once for that race.
3. A request after that race uses the published winner as the first sequential candidate when the winner is available.
4. Buffered races retain their existing behavior.
5. Existing tests pass, plus a focused regression test covers a winner that is not first in the configured order.
6. The regression test verifies that completion is published exactly once.
7. Requests between races retain `mode=race` in their request summary and response metadata while using the published preferred order.

Useful areas to inspect are `_race_request`, its background drain/finalization callbacks, `_finish_race`, and the preferred-provider selection in `proxy()`.
