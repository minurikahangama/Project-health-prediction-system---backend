"""
SQLAlchemy ORM models for all 6 database tables.

Tables:
  organisations      — tenant organisations
  users              — super_admin | org_admin | pm
  projects           — one PM owns one or more projects
  health_scores      — ML output stored as numbers only (GDPR)
  client_share_tokens — SHA-256 hashed share links
  gdpr_deletion_log  — audit trail of every raw-text deletion

GDPR compliance:
  - health_scores stores ONLY numeric values — never raw text
  - Raw email/transcript text is deleted immediately after processing
  - Every deletion is logged in gdpr_deletion_log with a SHA-256 fingerprint
  - Jira and Gmail tokens are AES-256 encrypted at rest
  - Client share tokens are stored as SHA-256 hash only
"""
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, ForeignKey, Text, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship
from app.utils.database import Base
from app.utils.time import utcnow
from sqlalchemy import UniqueConstraint


class Organisation(Base):
    __tablename__ = "organisations"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(255), unique=True, nullable=False)
    industry   = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    users    = relationship("User",    back_populates="organisation",
                            cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organisation",
                            cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Organisation id={self.id} name={self.name!r}>"


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    org_id        = Column(Integer,
                           ForeignKey("organisations.id", ondelete="CASCADE"),
                           nullable=False)
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)   # bcrypt cost 12
    role          = Column(String(50),  nullable=False)
    # role values: "super_admin" | "org_admin" | "pm"

    is_active              = Column(Boolean, default=True)
    force_password_change  = Column(Boolean, default=False)
    # force_password_change=True on first login for new accounts created by admins
    last_login             = Column(DateTime, nullable=True)
    address                = Column(Text, nullable=True)
    phone                  = Column(String(50), nullable=True)
    profile_image_url      = Column(String(500), nullable=True)
    created_at             = Column(DateTime, default=utcnow)

    # Relationships
    organisation = relationship("Organisation", back_populates="users")
    projects     = relationship("Project",      back_populates="pm")
    share_tokens = relationship("ClientShareToken",
                                foreign_keys="ClientShareToken.created_by_user_id",
                                back_populates="created_by")

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r} role={self.role!r}>"


class Project(Base):
    __tablename__ = "projects"

    id                    = Column(Integer, primary_key=True, index=True)
    org_id                = Column(Integer,
                                   ForeignKey("organisations.id", ondelete="CASCADE"),
                                   nullable=False)
    created_by_user_id    = Column(Integer,
                                   ForeignKey("users.id", ondelete="SET NULL"),
                                   nullable=True)
    name                  = Column(String(255), nullable=False)
    start_date            = Column(DateTime, nullable=False)
    deadline              = Column(DateTime, nullable=False)
    team_size             = Column(Integer, nullable=False)
    project_state         = Column(String(32), nullable=False, default="INITIATION")

    # Jira integration — tokens AES-256 encrypted at rest
    jira_url              = Column(String(500), nullable=True)
    encrypted_jira_token  = Column(Text, nullable=True)
    jira_email            = Column(String(255), nullable=True)

    # Gmail integration — token AES-256 encrypted at rest
    encrypted_gmail_token = Column(Text, nullable=True)
    gmail_account_email   = Column(String(255), nullable=True)
    gmail_filter_email    = Column(String(255), nullable=True)
    # Identifier which must appear in a message before it is attributed to this
    # project (for example "PHPS-42" or an internal project reference).
    gmail_project_identifier = Column(String(255), nullable=True)

    # Sync audit metadata shown in project detail pages.
    last_jira_synced_at  = Column(DateTime, nullable=True)
    last_jira_synced_by  = Column(String(255), nullable=True)
    last_email_synced_at = Column(DateTime, nullable=True)
    last_email_synced_by = Column(String(255), nullable=True)

    # Replaced after every Jira sync; contains Jira work metadata only.
    jira_capacity_snapshot = Column(JSON, nullable=True)
    jira_capacity_synced_at = Column(DateTime, nullable=True)
    # Replaced atomically after every Jira sync. This is evidence, not a
    # health contribution or prediction cache.
    jira_metrics_snapshot = Column(JSON, nullable=True)

    # Per-project RAG thresholds (FR-12)
    green_threshold       = Column(Float, default=70.0)
    red_threshold         = Column(Float, default=40.0)

    # PM note shown on client dashboard
    pm_note               = Column(Text, nullable=True)

    created_at            = Column(DateTime, default=utcnow)

    # Relationships
    organisation  = relationship("Organisation",    back_populates="projects")
    pm            = relationship("User",            back_populates="projects")
    health_scores = relationship("HealthScore",     back_populates="project",
                                 cascade="all, delete-orphan",
                                 order_by="HealthScore.recorded_at")
    share_tokens  = relationship("ClientShareToken", back_populates="project",
                                 cascade="all, delete-orphan")
    gdpr_logs     = relationship("GDPRDeletionLog",  back_populates="project",
                                 cascade="all, delete-orphan")
    transcripts   = relationship("TranscriptUpload", back_populates="project",
                                 cascade="all, delete-orphan")
    processed_emails = relationship("ProcessedEmail", back_populates="project",
                                    cascade="all, delete-orphan")
    team_members = relationship("ProjectTeamMember", back_populates="project",
                                cascade="all, delete-orphan")
    jira_evidence_snapshots = relationship("JiraEvidenceSnapshot", back_populates="project",
                                           cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project id={self.id} name={self.name!r}>"


class HealthScore(Base):
    """
    GDPR-critical table: stores ONLY numeric ML outputs.
    Raw text (emails, transcripts) is NEVER stored here.
    After processing, raw text is deleted and logged in GDPRDeletionLog.
    """
    __tablename__ = "health_scores"

    id               = Column(Integer, primary_key=True, index=True)
    project_id       = Column(Integer,
                               ForeignKey("projects.id", ondelete="CASCADE"),
                               nullable=False)

    # ML outputs — all numeric, never text (GDPR Article 5)
    health_score     = Column(Float, nullable=False)     # XGBoost output [0, 100]
    rag_status       = Column(String(10), nullable=False) # "GREEN" | "AMBER" | "RED"
    tone_score       = Column(Float, nullable=False)     # RoBERTa [-1.0, +1.0]
    urgency_flag     = Column(Integer, default=0)        # 0 or 1
    velocity_percent = Column(Float, nullable=False)     # Jira [0.0, 1.0]
    overdue_rate     = Column(Float, nullable=False)     # Jira [0.0, 1.0]
    bug_ratio        = Column(Float, nullable=False)     # Jira [0.0, 1.0]
    open_issues      = Column(Integer, default=0)
    open_bugs        = Column(Integer, default=0)
    divergence_flag  = Column(Integer, default=0)        # 0 or 1 — NOT Boolean
    # Numeric provenance permits a deleted transcript's observation to be
    # removed without retaining any of its raw text.
    analysis_source  = Column(String(20), nullable=True) # gmail|jira|transcript
    prediction_source = Column(String(64), nullable=True) # model provenance
    transcript_upload_id = Column(Integer, nullable=True, index=True)
    # Auditable snapshot of the fresh inputs and output explanations used for
    # this prediction. Never used as the next prediction's health input.
    feature_vector = Column(JSON, nullable=True)
    shap_explanation = Column(JSON, nullable=True)
    # Exact deduction values calculated from the live evidence at this point
    # in time. This is audit/history data only, never an input to a later run.
    deduction_snapshot = Column(JSON, nullable=True)

    recorded_at      = Column(DateTime, default=utcnow, index=True)

    # Relationship
    project = relationship("Project", back_populates="health_scores")

    def __repr__(self):
        return (f"<HealthScore id={self.id} project_id={self.project_id} "
                f"score={self.health_score} rag={self.rag_status!r}>")


class ProjectTeamMember(Base):
    """Latest Jira team-member and workload snapshot for one PHPS project."""
    __tablename__ = "project_team_members"
    __table_args__ = (UniqueConstraint("project_id", "jira_account_id", name="uq_project_jira_member"),)

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    jira_account_id = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    avatar_url = Column(String(1000), nullable=True)
    is_active_in_sprint = Column(Boolean, default=True, nullable=False)
    current_story_points = Column(Float, default=0.0, nullable=False)
    estimated_remaining_hours = Column(Float, default=0.0, nullable=False)
    remaining_tasks = Column(Integer, default=0, nullable=False)
    high_priority_tasks = Column(Integer, default=0, nullable=False)
    blocked_tasks = Column(Integer, default=0, nullable=False)
    workload_percentage = Column(Float, default=0.0, nullable=False)
    last_synced_at = Column(DateTime, default=utcnow, nullable=False, index=True)

    project = relationship("Project", back_populates="team_members")


class JiraEvidenceSnapshot(Base):
    """Immutable Jira state used only to calculate delivery trends."""
    __tablename__ = "jira_evidence_snapshots"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    metrics = Column(JSON, nullable=False)
    collected_at = Column(DateTime, default=utcnow, nullable=False, index=True)

    project = relationship("Project", back_populates="jira_evidence_snapshots")


class ClientShareToken(Base):
    """
    Secure client dashboard share links.
    Only the SHA-256 hash of the raw token is stored (NFR-14).
    The raw token is returned once on creation and never stored.
    """
    __tablename__ = "client_share_tokens"

    id                  = Column(Integer, primary_key=True, index=True)
    project_id          = Column(Integer,
                                  ForeignKey("projects.id", ondelete="CASCADE"),
                                  nullable=False)
    token_hash          = Column(String(64), unique=True, index=True, nullable=False)
    # SHA-256 produces a 64-character hex string
    expiry_date         = Column(DateTime, nullable=True)   # None = never expires
    created_by_user_id  = Column(Integer,
                                  ForeignKey("users.id", ondelete="SET NULL"),
                                  nullable=True)
    revoked             = Column(Boolean, default=False)
    last_accessed       = Column(DateTime, nullable=True)
    created_at          = Column(DateTime, default=utcnow)

    # Relationships
    project    = relationship("Project", back_populates="share_tokens")
    created_by = relationship("User",    back_populates="share_tokens",
                               foreign_keys=[created_by_user_id])

    def __repr__(self):
        return (f"<ClientShareToken id={self.id} project_id={self.project_id} "
                f"revoked={self.revoked}>")


class GDPRDeletionLog(Base):
    """
    Audit trail for GDPR Article 5 compliance.
    Every time raw text is processed and deleted, one row is written here.
    The confirmation_hash proves the raw text existed and was deleted.
    """
    __tablename__ = "gdpr_deletion_log"

    id                = Column(Integer, primary_key=True, index=True)
    event_type        = Column(String(100), nullable=False)
    # event_type: "Gmail Ingestion" | "Transcript Processing"
    project_id        = Column(Integer,
                                ForeignKey("projects.id", ondelete="CASCADE"),
                                nullable=False)
    confirmation_hash = Column(String(64), nullable=False)
    # SHA-256 hash of the raw text before deletion — proof of deletion
    deleted_at        = Column(DateTime, default=utcnow, index=True)

    # Relationship
    project = relationship("Project", back_populates="gdpr_logs")

    def __repr__(self):
        return (f"<GDPRDeletionLog id={self.id} event={self.event_type!r} "
                f"project_id={self.project_id}>")


class ProcessedEmail(Base):
    """Email metadata and numeric-only analysis evidence; bodies are never retained."""
    __tablename__ = "processed_emails"
    __table_args__ = (UniqueConstraint("project_id", "gmail_message_id", name="uq_processed_email"),)

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    gmail_message_id = Column(String(255), nullable=False)
    subject = Column(String(998), nullable=True)
    sender = Column(String(255), nullable=True)
    received_at = Column(DateTime, nullable=True)
    # These numeric observations make a future project-level prediction
    # reproducible without retaining the email body.
    tone_score = Column(Float, nullable=True)
    urgency_flag = Column(Integer, nullable=True)
    penalty = Column(Float, nullable=True)
    recovery = Column(Float, nullable=True)
    decay_weight = Column(Float, nullable=True)
    processed_at = Column(DateTime, default=utcnow, index=True)

    project = relationship("Project", back_populates="processed_emails")
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "gmail_message_id",
            name="uq_project_gmail_message",
        ),
    )

class TranscriptUpload(Base):
    """Metadata for a transcript retained at the project's request."""
    __tablename__ = "transcript_uploads"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    original_filename = Column(String(255), nullable=False)
    storage_filename = Column(String(255), nullable=False, unique=True)
    size_bytes = Column(Integer, nullable=False)
    content_sha256 = Column(String(64), nullable=False)
    # Persist only the derived values needed to recalculate after another
    # transcript is added or this one is removed.
    tone_score = Column(Float, nullable=True)
    urgency_flag = Column(Integer, nullable=True)
    penalty = Column(Float, nullable=True)
    recovery = Column(Float, nullable=True)
    decay_weight = Column(Float, nullable=True)
    uploaded_at = Column(DateTime, default=utcnow, index=True)

    project = relationship("Project", back_populates="transcripts")
