# RCA Generator

AI-powered Root Cause Analysis (RCA) Generator built using FastAPI, LangGraph, LangChain, and Google Gemini.

The system accepts structured issue-monitoring data and automatically generates a Root Cause Analysis report in Markdown, HTML, and PDF formats.

---

## Features

* AI-generated Root Cause Analysis
* LangGraph workflow orchestration
* Google Gemini integration
* HTML report generation
* PDF report generation using Playwright
* Audit logging
* Approval status tracking
* Secure PDF download endpoint
* API Key authentication
* Rate limiting
* LangSmith observability and tracing
* Cleanup policy for old generated files

---

## Tech Stack

### Backend

* FastAPI
* Uvicorn

### AI & Workflow

* LangChain
* LangGraph
* Google Gemini

### Templates & Documents

* Jinja2
* Playwright

### Security

* API Key Authentication
* SlowAPI Rate Limiting

### Monitoring

* LangSmith

### Testing

* Pytest
* Pytest Asyncio

---

## Project Structure

```text
rnd_rca_gen/
│
├── agent_root/
│   ├── chains.py
│   ├── graph.py
│   ├── models.py
│   ├── prompts.yaml
│   └── service.py
│
├── api/
│   ├── dependencies.py
│   ├── health.py
│   └── routes.py
│
├── common/
│   ├── rate_limit.py
│   └── utils.py
│
├── config/
│   ├── providers.py
│   └── settings.py
│
├── templates/
│   └── rca_template.html
│
├── static/
│   ├── rca.css
│   └── images/
│
├── outputs/
├── audit_logs/
├── tests/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rnd_rca_gen
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Playwright Chromium

Required for PDF generation.

```bash
playwright install chromium
```

---

## Environment Variables

Create a `.env` file.

```env
API_KEY=your_api_key

GOOGLE_API_KEY=your_google_api_key

GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
GEMINI_MAX_OUTPUT_TOKENS=8192
GEMINI_TOP_P=0.95
GEMINI_TOP_K=40

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rca-generator

RATE_LIMIT_DEFAULT=60/minute
RATE_LIMIT_RCA_GENERATION=10/minute
```

---

## Running the Application

Start the API server:

```bash
uvicorn app:app --reload
```

Default URL:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

---

### Secure Health Check

```http
GET /health/secure
```

Requires API Key.

---

### Gemini Health Check

```http
GET /health/gemini
```

Requires API Key.

---

### Generate RCA

```http
POST /api/generate-rca
```

Requires API Key.

Returns:

```json
{
  "issue_id": "EIL_2025000185",
  "approval_status": "Draft",
  "pdf_file_path": "/api/rca/EIL_2025000185/download"
}
```

---

### Download Generated PDF

```http
GET /api/rca/{issue_id}/download
```

Requires API Key.

Example:

```http
GET /api/rca/EIL_2025000185/download
```

Returns:

```text
application/pdf
```

---

## Audit Logging

Each successful RCA generation creates a JSON audit record.

Location:

```text
audit_logs/
```

Example:

```json
{
  "issue_id": "EIL_2025000185",
  "client": "AMC",
  "generated_at": "2026-06-15T10:30:00Z",
  "generated_by": "RCA Generator",
  "pdf_file_path": "outputs/EIL_2025000185_rca.pdf",
  "status": "Draft"
}
```

---

## Approval Status

Supported statuses:

```text
Draft
For Review
Approved
Rejected
```

Default status:

```text
Draft
```

---

## Cleanup Policy

Generated files older than 30 days are automatically removed.

Affected folder:

```text
outputs/
```

Supported file types:

```text
.html
.pdf
```

---

## Running Tests

Run all tests:

```bash
pytest
```

Run route tests:

```bash
pytest tests/test_routes.py
```

Expected:

```text
5 passed
```

---

## Deployment Notes

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Playwright browser:

```bash
playwright install chromium
```

Without Chromium installation, PDF generation will fail.

---

## Future Enhancements

* Approval Workflow UI
* Email Notifications
* PDF Versioning
* Audit Log API
* Database Integration
* GitHub Pull Request Automation

---

## Author

Research and Development Team

Direc Business Technologies Inc.
