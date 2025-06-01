# -*- coding: utf-8 -*-
"""Functions for loading and managing NLP models."""

import spacy
from sentence_transformers import SentenceTransformer
import torch
import os
from .config import SPACY_MODEL_NAME, SENTENCE_MODEL_NAME, DEFAULT_DEVICE

# Global variables to hold loaded models (simple caching)
_nlp = None
_sentence_model = None
_device = None

def get_device():
    """Determines the appropriate device (CUDA or CPU)."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = DEFAULT_DEVICE
        print(f"Using device: {_device}")
    return _device

def load_spacy_model(model_name=SPACY_MODEL_NAME):
    """Loads the spaCy NLP model. Downloads if not available."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(model_name)
            print(f"spaCy model 	'{model_name}' loaded successfully.")
        except OSError:
            print(f"spaCy model 	'{model_name}' not found. Downloading...")
            try:
                spacy.cli.download(model_name)
                _nlp = spacy.load(model_name)
                print(f"spaCy model 	'{model_name}' downloaded and loaded successfully.")
            except Exception as e:
                print(f"Error downloading or loading spaCy model 	'{model_name}': {e}")
                _nlp = None # Ensure it remains None if loading fails
    return _nlp

def load_sentence_transformer_model(model_name=SENTENCE_MODEL_NAME):
    """Loads the Sentence Transformer model."""
    global _sentence_model
    if _sentence_model is None:
        device = get_device()
        try:
            _sentence_model = SentenceTransformer(model_name, device=device)
            print(f"Sentence Transformer model 	'{model_name}' loaded successfully on {device}.")
        except Exception as e:
            print(f"Error loading Sentence Transformer model 	'{model_name}': {e}")
            _sentence_model = None # Ensure it remains None if loading fails
    return _sentence_model

# Pre-load models on import if desired (optional)
# load_spacy_model()
# load_sentence_transformer_model()

