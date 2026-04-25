"""Prometheus registry + OTel tracer factory.

A single shared CollectorRegistry per process (so /metrics dumps the union
of all metric definitions). Tracer factory returns a no-op tracer until
configure_tracing() wires up an OTLP exporter — Phase 6/7.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .config import get_settings

REGISTRY = CollectorRegistry()


def counter(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Counter:
    return Counter(name, doc, labelnames, registry=REGISTRY)


def gauge(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Gauge:
    return Gauge(name, doc, labelnames, registry=REGISTRY)


def histogram(
    name: str,
    doc: str,
    labelnames: tuple[str, ...] = (),
    buckets: tuple[float, ...] | None = None,
) -> Histogram:
    if buckets is None:
        return Histogram(name, doc, labelnames, registry=REGISTRY)
    return Histogram(name, doc, labelnames, buckets=buckets, registry=REGISTRY)


_provider: TracerProvider | None = None


def configure_tracing() -> TracerProvider:
    """Install a TracerProvider tagged with service metadata. Idempotent."""
    global _provider
    if _provider is not None:
        return _provider
    settings = get_settings()
    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.version,
            "deployment.environment": settings.app_env,
        }
    )
    _provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "sam3") -> Tracer:
    if _provider is None:
        configure_tracing()
    return trace.get_tracer(name)
