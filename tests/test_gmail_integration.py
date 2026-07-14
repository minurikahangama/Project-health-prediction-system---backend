import os
import unittest
from email.message import EmailMessage
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.api import pipeline
from app.services import gmail_ingestion


class _Mailbox:
    def __init__(self, raw_message: bytes):
        self.raw_message = raw_message
        self.selected = []
        self.search_args = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def login(self, username, password):
        self.login_args = (username, password)
        return "OK", []

    def select(self, folder, readonly=True):
        self.selected.append((folder, readonly))
        return "OK", []

    def search(self, _charset, *criteria):
        self.search_args = criteria
        return "OK", [b"1"]

    def fetch(self, uid, _query):
        return "OK", [(b"1 (RFC822)", self.raw_message)]


class _Query:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return None


class _Session:
    def __init__(self):
        self.added = []
        self.committed = False

    def query(self, *_args):
        return _Query()

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True


class GmailIntegrationTests(unittest.TestCase):
    def test_fetches_plain_text_messages_from_the_monitored_address(self):
        message = EmailMessage()
        message["Message-ID"] = "<message-1@example.com>"
        message["Subject"] = "PHPS-4 sprint update"
        message["From"] = "client@example.com"
        message["To"] = "team@example.com"
        message["Date"] = "Mon, 13 Jul 2026 10:00:00 +0000"
        message.set_content("The PHPS-4 sprint update is delayed.")
        mailbox = _Mailbox(message.as_bytes())

        with patch.object(gmail_ingestion, "decrypt_token", return_value="app-password"), patch.object(
            gmail_ingestion.imaplib, "IMAP4_SSL", return_value=mailbox
        ):
            rows = gmail_ingestion.fetch_messages(
                sender="client@example.com",
                account_email="team@example.com",
                encrypted_token="encrypted",
            )

        self.assertEqual(rows[0]["id"], "<message-1@example.com>")
        self.assertEqual(rows[0]["subject"], "PHPS-4 sprint update")
        self.assertIn("delayed", rows[0]["body"])
        self.assertIn("FROM", mailbox.search_args)
        self.assertIn("TO", mailbox.search_args)

    def test_ingestion_filters_by_identifier_and_records_only_matching_messages(self):
        db = _Session()
        project = SimpleNamespace(id=4, gmail_project_identifier="PHPS-4")
        messages = [
            pipeline.GmailMessage(id="match", subject="PHPS-4 status", body="Sprint is delayed", sender="client@example.com"),
            pipeline.GmailMessage(id="other", subject="Other project", body="Ignore this", sender="client@example.com"),
        ]
        with patch.object(pipeline, "process_and_delete", return_value="Sprint is delayed") as process, patch.object(
            pipeline, "_run_pipeline_and_save", return_value={"status": "ok", "project_id": 4}
        ) as run_pipeline:
            result = pipeline._ingest_gmail_messages(db, project, messages)

        self.assertEqual(result["processed_messages"], 1)
        self.assertTrue(db.committed)
        self.assertEqual(db.added[0].gmail_message_id, "match")
        self.assertEqual(db.added[0].subject, "PHPS-4 status")
        self.assertEqual(process.call_args.kwargs["event_type"], "Gmail Ingestion")
        self.assertIsNone(run_pipeline.call_args.kwargs["jira_signals"])

    def test_manual_sync_requires_saved_gmail_credentials(self):
        project = SimpleNamespace(
            gmail_account_email=None,
            encrypted_gmail_token=None,
            gmail_filter_email="client@example.com",
            gmail_project_identifier="PHPS-4",
        )
        with self.assertRaises(gmail_ingestion.GmailIngestionError):
            pipeline.sync_gmail_project(object(), project)

    def test_ingestion_deduplicates_repeated_message_ids_in_one_sync(self):
        db = _Session()
        project = SimpleNamespace(id=4, gmail_project_identifier="PHPS-4")
        message = pipeline.GmailMessage(
            id="duplicate", subject="PHPS-4 status", body="Sprint is delayed"
        )
        with patch.object(pipeline, "process_and_delete", return_value="Sprint is delayed"), patch.object(
            pipeline, "_run_pipeline_and_save", return_value={"status": "ok", "project_id": 4}
        ) as run_pipeline:
            result = pipeline._ingest_gmail_messages(db, project, [message, message])

        self.assertEqual(result["processed_messages"], 1)
        self.assertEqual(result["duplicate_messages"], 1)
        self.assertEqual(len(db.added), 1)
        run_pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
