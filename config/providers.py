from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_openai import (
    ChatOpenAI
)

from config.settings import settings


# PRIMARY MODEL
openai_llm = ChatOpenAI(
    model=settings.OPENAI_CHAT_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)


# BACKUP MODEL
gemini_summary_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_SUMMARY_MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)

gemini_mom_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MOM_MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)