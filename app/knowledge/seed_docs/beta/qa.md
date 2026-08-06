# Beta Project — QA Document

## Payment Acceptance Criteria
A successful card payment records the amount, currency, and merchant reference.
A failed payment is retried up to three times and then marked as declined.

## Refund Test Scenarios
A refund can be issued only against a settled transaction. A refund greater than
the original transaction amount is rejected by BetaLedger.
