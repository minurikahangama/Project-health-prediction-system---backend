"""Transcript uploads must always create an analysis event."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.api import pipeline


class _Session:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, item):
        self.added.append(item)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def delete(self, item):
        self.added.remove(item)


def _upload(filename: str, content: bytes) -> UploadFile:
    handle = tempfile.SpooledTemporaryFile()
    handle.write(content)
    handle.seek(0)
    return UploadFile(file=handle, filename=filename, headers=Headers({"content-type": "text/plain"}))


class TranscriptUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_content_is_saved_and_scored_for_each_upload(self):
        db = _Session()
        project = SimpleNamespace(id=7)
        user = SimpleNamespace(id=4)
        result = {"status": "ok", "health_score": 75.0}
        first_upload = _upload("meeting.txt", b"All goals are complete")
        second_upload = _upload("meeting.txt", b"All goals are complete")

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, "TRANSCRIPT_DIR", Path(directory)), \
             patch.object(pipeline, "_project_for_user", return_value=project), \
             patch.object(pipeline, "process_and_delete", return_value="All goals are complete"), \
             patch.object(pipeline, "_run_pipeline_and_save", return_value=result.copy()) as score:
            try:
                first = await pipeline.upload_transcript(7, first_upload, db, user)
                second = await pipeline.upload_transcript(7, second_upload, db, user)
            finally:
                await first_upload.close()
                await second_upload.close()

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual(len(db.added), 2)
        self.assertEqual(score.call_count, 2)
        self.assertNotEqual(db.added[0].storage_filename, db.added[1].storage_filename)


if __name__ == "__main__":
    unittest.main()
