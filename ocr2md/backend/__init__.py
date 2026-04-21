"""OCR Backend module.

This module contains OCR backend implementations for different models.
"""

from .base import BaseBackend
from .glm_ocr_backend import GLMOCRBackend

__all__ = ["BaseBackend", "GLMOCRBackend"]
