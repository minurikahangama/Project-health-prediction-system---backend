# TechCorp Platform Redesign — Requirement Document

## Overview
The TechCorp Platform Redesign modernises the existing customer platform with a
new React front end, a consolidated REST API, and a migration of legacy account
data. The goal is to improve performance, accessibility, and maintainability
without disrupting existing customers.

## Authentication and SSO
Users shall sign in with their corporate email through single sign-on (SSO).
A successful sign-in shall issue a session token that expires after sixty
minutes of inactivity. Users without SSO access shall be denied entry and shown
a clear message.

## Customer Dashboard
The redesigned dashboard shall display each customer's account summary, recent
activity, and open support tickets. All dashboard metrics shall load within two
seconds on a standard broadband connection.

## Data Migration
Legacy account records shall be migrated to the new schema in nightly batches.
Each migrated record shall preserve its original account identifier. Records
that fail validation shall be logged and skipped, never partially imported.

## Performance and Accessibility
Every page shall meet WCAG 2.1 AA accessibility standards. The platform shall
support at least one thousand concurrent users without response times exceeding
two seconds.
