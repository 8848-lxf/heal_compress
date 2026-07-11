"""Typed exceptions raised by the formal tracing package."""

from __future__ import annotations


class TraceError(RuntimeError):
    """Base class for failures that make a trace unsafe to consume."""


class UnsupportedOperationError(TraceError):
    """A channel-changing operation has no proven channel mapping."""


class AmbiguousDependencyError(TraceError):
    """A channel dependency has more than one incompatible interpretation."""


class TraceSerializationError(TraceError):
    """A serialized trace cannot be read or does not satisfy its schema."""

