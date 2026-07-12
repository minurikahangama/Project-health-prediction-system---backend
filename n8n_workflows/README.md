# PHPS n8n workflows

Import the two JSON templates into n8n, then complete the placeholders before
activating them.

1. In the backend `.env`, set `N8N_WEBHOOK_SECRET` to a new random value and
   restart the API.
2. Replace the template API URL with the backend address reachable from n8n.
   Use `http://localhost:8000` when both run on Windows; use
   `http://host.docker.internal:8000` when n8n runs in Docker on Windows.
3. Set the same value in every `X-N8N-Secret` header.
4. Select your Gmail OAuth2 credential in the Gmail node. Do not put Gmail or
   Jira tokens in these workflow files.
5. Configure each project with a complete Jira Cloud URL, e.g.
   `https://your-site.atlassian.net/jira/software/projects/PHPS`.

Both workflows discover all configured PHPS projects on every execution. A new
project automatically joins the next run as soon as its Gmail filter address,
Gmail project identifier, and/or Jira settings are saved. The identifier must
appear in the email subject or body. The Jira template uses the cron expression
`0 0 9 * * 1`: Monday at 09:00 Asia/Colombo time. Test each workflow manually
once before activating its schedule.

The backend deduplicates Gmail message IDs, so testing or retrying the Gmail
workflow does not reprocess an already accepted message. Jira API credentials
remain encrypted in the PHPS database; the Jira n8n workflow only triggers a
sync for each configured project.
