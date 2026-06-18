# AI Meeting Notes Summarizer (MOM Generator)

## Overview

The AI Meeting Notes Summarizer is a Supervisor-Orchestrated Multi-Agent AI system that automatically converts meeting audio recordings into professional Minutes of Meeting (MOM) documents.

The system leverages OpenAI and Google Gemini models through a multi-layer fallback architecture to ensure reliability, accuracy, and business-ready document generation.

Generated outputs include:

* Meeting Transcript (.txt)
* Minutes of Meeting (.pdf)
* Minutes of Meeting (.docx) (Optional)

---

# Features

## Audio Transcription

Converts meeting audio into text transcripts.

Primary Model:

* GPT-4o Mini Transcribe

Fallback Model:

* Gemini 3.5 Flash

---

## Structured Meeting Summary

Extracts:

* Subject Title
* Attendees
* Owners
* Key Discussion Points
* Decisions Made
* Important Information Discussed
* Detailed Summary

Primary Model:

* GPT-5.4 Mini

Fallback Model:

* Gemini 2.5 Flash Lite

---

## MOM Generation

Generates a professional Minutes of Meeting document using:

* Structured Summary (Primary Source)
* Full Transcript (Validation Source)

Primary Model:

* GPT-5.4 Mini

Fallback Model:

* Gemini 2.5 Flash

---

## Supervisor Agent

Acts as the Mother Agent.

Responsibilities:

* Orchestrates workflow
* Routes execution
* Validates outputs
* Detects missing information
* Checks consistency
* Reviews generated MOM
* Approves or rejects final output

---

## Output Formats

Supported Outputs:

* TXT Transcript
* PDF MOM
* DOCX MOM (Optional)

---

# System Architecture

Audio File
↓
Supervisor Agent
↓
Transcriber Agent
↓
Supervisor Validation
↓
Summarizer Agent
↓
Supervisor Validation
↓
MOM Generator Agent
↓
Supervisor Final Review
↓
HTML Generation
↓
PDF Generation
↓
Optional DOCX Generation
↓
Download List

---

# Multi-Agent Architecture

## Supervisor Agent

Purpose:

Central orchestrator of the system.

Responsibilities:

* Workflow routing
* Validation
* Quality control
* Consistency checking
* Final approval

---

## Transcriber Agent

Purpose:

Convert meeting audio into an accurate transcript.

Responsibilities:

* Preserve meeting content
* Maintain speaker context when possible
* Avoid summarization

Output:

Transcript TXT

---

## Summarizer Agent

Purpose:

Generate a structured summary from the transcript.

Output:

SUBJECT TITLE

ATTENDEES

OWNERS

KEY DISCUSSION POINTS

DECISIONS MADE

IMPORTANT INFORMATION DISCUSSED

DETAILED SUMMARY

---

## MOM Generator Agent

Purpose:

Generate a professional Minutes of Meeting document.

Behavior:

1. Uses Structured Summary first.
2. If information is unclear:

   * Validates against Full Transcript.
3. Produces final MOM output.

---

# Technology Stack

Backend:

* FastAPI

AI Framework:

* LangChain
* LangGraph

Models:

* OpenAI GPT-4o Mini Transcribe
* OpenAI GPT-5.4 Mini
* Gemini 3.5 Flash
* Gemini 2.5 Flash
* Gemini 2.5 Flash Lite

Observability:

* LangSmith

Validation:

* Pydantic

Document Generation:

* HTML
* PDF
* DOCX

Security:

* API Key Authentication

---

# Project Structure

```text
rnd_mom_gen/

├── agent_1_sum/
│   ├── chains.py
│   ├── graph.py
│   ├── prompts.yaml
│   ├── models.py
│   ├── output/
│   │   ├── transcripts/
│   │   ├── summaries/
│   │   ├── html/
│   │   ├── pdf/
│   │   └── docx/
│
├── api/
│   ├── routes.py
│   ├── dependencies.py
│
├── config/
│   ├── providers.py
│   ├── settings.py
│
├── uploads/
│
├── app.py
│
└── requirements.txt
```

# Environment Variables

Create a .env file:

```env
API_KEY=rnd_mom_xxxxxxxx
API_KEY_HEADER=X-API-Key

OPENAI_API_KEY=your_openai_key

OPENAI_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe
OPENAI_CHAT_MODEL=gpt-5.4-mini

GEMINI_API_KEY=your_gemini_key

GEMINI_TRANSCRIBE_MODEL=gemini-3.5-flash
GEMINI_SUMMARY_MODEL=gemini-2.5-flash-lite
GEMINI_MOM_MODEL=gemini-2.5-flash

LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=rnd-mom-gen
LANGSMITH_TRACING=true
```

# Security

API Authentication is required.

Header:

```text
X-API-Key
```

Example:

```text
X-API-Key: rnd_mom_xxxxxxxx
```

Authentication uses:

* APIKeyHeader
* HMAC Comparison
* Environment Variables

---

# API Endpoints

## Generate MOM

POST

```text
/api/generate-mom
```

Input:

* Audio File
* Generate DOCX (optional)

Output:

```json
{
  "message": "MOM generated successfully",
  "transcript_path": "...",
  "pdf_path": "...",
  "docx_path": "...",
  "downloads": [...],
  "documents": [...]
}
```

---

## Download Transcript

GET

```text
/api/download/txt/{filename}
```

---

## Download PDF

GET

```text
/api/download/pdf/{filename}
```

---

## Download DOCX

GET

```text
/api/download/docx/{filename}
```

---

## Health Check

GET

```text
/health
```

---

## Secure Health Check

GET

```text
/health/secure
```

---

# Output Files

Generated Transcript:

```text
agent_1_sum/output/transcripts/
```

Generated HTML:

```text
agent_1_sum/output/html/
```

Generated PDF:

```text
agent_1_sum/output/pdf/
```

Generated DOCX:

```text
agent_1_sum/output/docx/
```

---

# Supervisor Validation Flow

Transcript
↓
Summary Validation
↓
MOM Validation
↓
Consistency Checking
↓
Approval Decision

Possible Results:

* Approved
* Rejected with Review Notes

---

# Known Limitations

Current Development Environment:

* OpenAI requires active billing/quota.
* Gemini free-tier may experience:

  * Quota limits
  * High-demand delays
  * Temporary service unavailability (503).

Fallback mechanisms are implemented to improve reliability.

---

# Author

Research and Development Project

AI Meeting Notes Summarizer

Supervisor-Orchestrated Multi-Agent Architecture

FastAPI + LangGraph + LangChain + OpenAI + Gemini
