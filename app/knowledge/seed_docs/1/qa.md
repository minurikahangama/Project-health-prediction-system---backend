# TechCorp Platform Redesign — QA Document

## SSO Login Acceptance Criteria
A user with valid corporate SSO credentials is signed in and redirected to the
dashboard. A user without SSO access is denied and shown an error message. The
session expires after sixty minutes of inactivity.

## Dashboard Test Scenarios
The customer dashboard displays the account summary, recent activity, and open
support tickets. All dashboard metrics load within two seconds. An account with
no activity shows an empty-state message instead of an error.

## Migration Test Scenarios
Migrated account records keep their original account identifier. A record that
fails validation is logged and skipped, and does not appear partially imported
in the new schema.
