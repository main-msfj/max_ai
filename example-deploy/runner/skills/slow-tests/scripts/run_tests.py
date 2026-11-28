"""A test suite that takes its time: twelve made-up tests, one of them failing."""

import sys
import time

TESTS = [
    "test_login", "test_logout", "test_signup", "test_cart_add", "test_cart_remove",
    "test_checkout", "test_refund_rounding", "test_invoice_pdf", "test_email_receipt",
    "test_search", "test_export_csv", "test_admin_roles",
]
FAILS = {"test_refund_rounding": "expected 10.05, got 10.04"}


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"collected {len(TESTS)} tests", flush=True)
    for number, name in enumerate(TESTS, 1):
        time.sleep(seconds)
        result = f"FAILED ({FAILS[name]})" if name in FAILS else "PASSED"
        print(f"[{number:>2}/{len(TESTS)}] {name} {result}", flush=True)
    print(f"{len(TESTS) - len(FAILS)} passed, {len(FAILS)} failed", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
