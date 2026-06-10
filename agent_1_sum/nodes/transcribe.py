import whisper

from common.logging import logger
from common.exceptions import TranscriptionException


def transcribe_audio(
    audio_path: str
) -> str:
    try:
        logger.info(
            f"Starting transcription for: {audio_path}"
        )

        model = whisper.load_model("base")

        result = model.transcribe(
            audio_path
        )

        transcript = result.get(
            "text",
            ""
        )

        logger.info(
            "Transcription completed successfully"
        )

        return transcript

    except Exception as error:
        logger.error(
            f"Transcription failed: {error}"
        )

        raise TranscriptionException(
            "Failed to transcribe audio file"
        ) from error