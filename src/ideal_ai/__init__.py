"""
ideal_ai - Universal LLM Connector Package

A professional, production-ready Python package for unified access to multiple LLM providers.
Supports text generation, vision, audio transcription, speech synthesis, image and video generation.
"""

from ideal_ai.connector import IdealUniversalLLMConnector, IdealSmolagentsWrapper

__version__ = "0.2.0"
__author__ = "Gilles Blanchet"
__license__ = "Apache-2.0"

__all__ = [
    "IdealUniversalLLMConnector",
    "IdealSmolagentsWrapper",
]