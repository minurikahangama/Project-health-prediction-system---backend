# Alpha Project — Requirement Document

## User Authentication
The AlphaVault application shall authenticate users with an email address and a
password. Passwords shall be at least eight characters long and contain one
uppercase letter, one number, and one special character. After five failed
login attempts within fifteen minutes the account shall be temporarily locked.

## Session Management
A successful login shall issue a JWT access token that expires after sixty
minutes of inactivity. Users shall be redirected to a role-specific dashboard
after signing in.

## Password Reset
Users shall be able to request a password reset link by email. The reset link
shall expire after thirty minutes and may be used only once.
