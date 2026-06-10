from config.settings import settings


def get_google_api_key() -> str:
    return settings.GOOGLE_API_KEY