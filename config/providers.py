from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from config.settings import settings


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0
)