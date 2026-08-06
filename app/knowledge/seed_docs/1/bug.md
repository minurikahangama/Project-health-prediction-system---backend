# TechCorp Platform Redesign — Bug Report

## TECH-301 Session does not expire
The session token remains valid beyond sixty minutes of inactivity instead of
expiring, which violates the Authentication requirement. Severity: High.
Status: Open. Assignee: Sarah Johnson.

## TECH-302 Dashboard slow to load
The customer dashboard takes more than five seconds to load account metrics,
exceeding the two-second performance requirement. Severity: Medium. Status: In
Progress. Assignee: David Lee.

## TECH-303 Migration skips valid records
The nightly migration skips some valid account records and logs them as failed
validation, so a small number of accounts are missing from the new schema.
Severity: High. Status: Open. Assignee: Priya Raman.
