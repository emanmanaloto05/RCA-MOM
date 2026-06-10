from dotenv import load_dotenv
import os

load_dotenv()

print("Google Key:", os.getenv("GOOGLE_API_KEY"))
print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))