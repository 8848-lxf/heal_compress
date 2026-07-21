"""Experimental model-family-specific deployment components.

This package deliberately does not register a search runner.  The H800
Transformer study freezes structure and exposes only audited quantization
profiles; the existing Pyramid search dispatch remains the sole production
search entry point.
"""
