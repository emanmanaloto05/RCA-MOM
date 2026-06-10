class InvalidAudioFileException(Exception):
    """Raised when uploaded file is not a valid audio file."""
    pass


class TranscriptionException(Exception):
    """Raised when audio transcription fails."""
    pass


class MOMGenerationException(Exception):
    """Raised when MOM generation fails."""
    pass


class PDFGenerationException(Exception):
    """Raised when PDF creation fails."""
    pass