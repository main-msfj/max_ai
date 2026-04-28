# Severity Level Definitions

Use these definitions when classifying a bug. Pick the highest level
that applies — if a bug fits both "high" and "medium", call it high.

## Critical

The bug causes one or more of:

- Data loss or corruption.
- Security breach (unauthorized access, leaked credentials, exposed PII).
- Production system unavailable for all or most users.
- Financial loss (failed payments, double charges, incorrect billing).
- Regulatory violation (privacy, accessibility, jurisdiction-specific laws).

**Examples:**
- "Login fails for every user since the deploy."
- "User passwords are appearing in server logs."
- "Database returned wrong amounts on 200 invoices."

## High

The bug significantly impairs core functionality but doesn't meet critical:

- Major feature broken for a meaningful subset of users.
- Frequent crashes, 500s, or timeouts on key flows.
- Recent regression that blocks expected behavior.
- Performance degradation that makes the product hard to use.

**Examples:**
- "Search results are empty for queries with apostrophes."
- "Mobile app crashes on the checkout screen for iOS 17 users."
- "Dashboard takes 40 seconds to load since yesterday."

## Medium

The bug causes friction but workarounds exist:

- Edge cases with predictable triggers.
- Intermittent issues without clear cause.
- Non-critical features misbehaving.
- Inconvenient but tolerable UX.

**Examples:**
- "Date picker occasionally selects the wrong day in DST transitions."
- "Pagination breaks when there are exactly 100 items."
- "Email subject line gets truncated in some clients."

## Low

The bug is cosmetic or has no functional impact:

- Typos, alignment, color, spacing.
- Wording that's confusing but not wrong.
- Outdated copy or links to deprecated content.
- Minor UI inconsistencies that don't block any task.

**Examples:**
- "Button label says 'Sumbit' instead of 'Submit'."
- "Modal close icon is 1px off-center on Firefox."
- "Help text says 'click here' instead of describing the link."