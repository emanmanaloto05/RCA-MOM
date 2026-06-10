# test_settings.py

from config.settings import settings

print("Project:", settings.LANGSMITH_PROJECT)
print("Google Key Loaded:", bool(settings.GOOGLE_API_KEY))