# PR Review Checklist

Walk through each section. Mark items as ✅ (passes), ⚠️ (needs attention),
or ❌ (blocker).

## Security

- [ ] Are user inputs validated before use?
- [ ] Are SQL queries parameterized (no string concatenation)?
- [ ] Are secrets read from environment or vault, never hardcoded?
- [ ] Are auth checks present on every protected endpoint?
- [ ] Is sensitive data logged anywhere it shouldn't be?
- [ ] Are dependencies updated and free of known CVEs?

## Tests

- [ ] Are new code paths covered by tests?
- [ ] Do edge cases have explicit test cases (empty, null, max, boundary)?
- [ ] Are flaky or skipped tests addressed or annotated?
- [ ] Do integration tests cover the new flows end-to-end?

## Error handling

- [ ] Are errors handled at the right layer (not swallowed silently)?
- [ ] Are user-facing error messages clear without leaking internals?
- [ ] Is there a fallback or graceful degradation path?

## Performance

- [ ] Any obvious N+1 queries or unbounded loops?
- [ ] Are large payloads streamed or paginated?
- [ ] Are caches invalidated when data changes?

## Observability

- [ ] Are key operations logged at the right level?
- [ ] Are metrics or traces emitted for critical paths?
- [ ] Are alerts configured for new failure modes?

## Documentation

- [ ] Is the README updated if the public API changed?
- [ ] Are new endpoints documented?
- [ ] Are migration steps noted for breaking changes?