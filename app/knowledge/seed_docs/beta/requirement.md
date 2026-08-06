# Beta Project — Requirement Document

## Payment Processing
The BetaLedger service shall process card payments through the payment gateway.
Each transaction shall record an amount, a currency, and a merchant reference.
Failed payments shall be retried up to three times before being marked as
declined.

## Refunds
A refund shall be issued only against a settled transaction. A refund amount
shall never exceed the original transaction amount.

## Reconciliation
The BetaLedger service shall reconcile settled transactions against the gateway
report every night and flag any mismatched records for review.
