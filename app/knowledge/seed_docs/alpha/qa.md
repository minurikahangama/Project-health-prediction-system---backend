# Alpha Project — QA Document

## Login Acceptance Criteria
A user can log in with valid credentials and is redirected to their dashboard.
Invalid credentials display an appropriate error message. After five failed
attempts the AlphaVault account is locked for fifteen minutes.

## Password Reset Test Scenarios
A reset link sent by email expires after thirty minutes. A reset link cannot be
reused once the password has been changed. An expired reset link shows a clear
"link expired" message.
