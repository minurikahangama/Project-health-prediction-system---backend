<<<<<<< HEAD
# PHPS Backend — Setup Guide

## Quick Start (5 steps)

### Step 1 — Install PostgreSQL
Download from: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
During install: remember the password you set for the `postgres` user.

### Step 2 — Create the database
Open Command Prompt and run:
```
psql -U postgres
CREATE DATABASE phps_db;
CREATE USER phps_user WITH ENCRYPTED PASSWORD 'YourPassword123!';
GRANT ALL PRIVILEGES ON DATABASE phps_db TO phps_user;
\c phps_db
GRANT ALL ON SCHEMA public TO phps_user;
\q
```

### Step 3 — Create virtual environment and install packages
```
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 4 — Configure environment
```
copy .env.example .env
```
Then edit `.env`:
- Set `DATABASE_URL` with your PostgreSQL password
- Generate `SECRET_KEY`:  `python -c "import secrets; print(secrets.token_hex(32))"`
- Generate `ENCRYPTION_KEY`:  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`

### Step 5 — Run migrations and seed
```
alembic init migrations          # only if migrations folder is empty
alembic upgrade head             # creates all 6 tables
python seed_data.py              # creates initial users and sample data
```

### Step 6 — Start the server
```
uvicorn app.main:app --reload --port 8000
```
Open: http://localhost:8000/docs

---

## Login credentials (after seeding)
| Role        | Email                      | Password      |
|-------------|----------------------------|---------------|
| Super Admin | admin@phps.com             | Admin@123     |
| Org Admin   | orgadmin@techcorp.com      | OrgAdmin@123  |
| PM          | pm@techcorp.com            | PM@1234567    |

---

## Project structure
```
backend/
├── app/
│   ├── api/
│   │   ├── auth.py          ← JWT login, bcrypt, get_current_user
│   │   ├── projects.py      ← project CRUD + health score history
│   │   ├── pipeline.py      ← n8n ingest endpoints + transcript upload
│   │   ├── client.py        ← share token generation + public client view
│   │   └── admin.py         ← org management, user creation, audit
│   ├── models/
│   │   └── models.py        ← all 6 SQLAlchemy tables
│   ├── ml/
│   │   ├── preprocessor.py  ← spaCy PII anonymisation (GDPR)
│   │   ├── sentiment.py     ← RoBERTa tone scorer + urgency detection
│   │   ├── jira_signals.py  ← velocity, overdue, bug ratio from Jira
│   │   ├── scorer.py        ← XGBoost fusion + divergence detection
│   │   ├── inference/       ← put xgb_model.json and roberta_phps/ here
│   │   └── training/        ← fine_tune_roberta.py, train_xgboost.py
│   ├── utils/
│   │   ├── database.py      ← SQLAlchemy engine, session, Base
│   │   └── encryption.py    ← AES-256 Fernet encrypt/decrypt
│   └── main.py              ← FastAPI app, all routers, WebSocket
├── migrations/
│   └── env.py               ← Alembic config pointing to your models
├── requirements.txt
├── seed_data.py
└── .env.example             ← copy to .env and fill in your values
```

---

## ML Models

### RoBERTa (fine-tuned)
Place trained model files in: `app/ml/inference/roberta_phps/`
Required files: config.json, pytorch_model.bin (or model.safetensors),
tokenizer.json, tokenizer_config.json, vocab.json, merges.txt

If not present, the system uses `cardiffnlp/twitter-roberta-base-sentiment-3`
as a fallback (downloads automatically on first run).

### XGBoost
Place trained model at: `app/ml/inference/xgb_model.json`
Train with: `python app/ml/training/train_xgboost.py`

If not present, a rule-based scoring formula is used as fallback.

---

## API Endpoints Summary
| Method | Path | Description |
|--------|------|-------------|
| POST | /auth/login | Login → JWT token |
| POST | /auth/change-password | First-login password change |
| GET | /projects/ | List projects |
| POST | /projects/ | Create project |
| GET | /projects/{id}/health-score | Latest score + history |
| PATCH | /projects/{id}/pm-note | Update client-facing note |
| POST | /projects/test-jira | Test Jira connection |
| POST | /pipeline/ingest-gmail | n8n: process Gmail messages |
| POST | /pipeline/ingest-jira | n8n: sync Jira signals |
| POST | /pipeline/upload-transcript | PM: upload meeting transcript |
| POST | /client/projects/{id}/share-link | Generate client share link |
| GET | /client/{token} | Public client dashboard data |
| GET | /admin/organisations | List all orgs (super_admin) |
| POST | /admin/users | Create PM or Org Admin account |
| GET | /admin/export-research | Download anonymised research CSV |
| WS | /ws/{project_id} | Real-time health score updates |
=======
# phps-research
Software project health prediction using communication sentiments and jira delivery signals.This focuses on predicting software project failures earlt before they escalate using AI by combining human and technical factors
>>>>>>> 989d2d25c773dd4bd6d60235f6016fceba386093
