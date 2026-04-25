from prometheus_client import generate_latest

from packages.core import telemetry


def test_counter_increment_and_scrape():
    c = telemetry.counter("test_counter_total", "doc", ("kind",))
    c.labels(kind="a").inc()
    c.labels(kind="a").inc(2)
    text = generate_latest(telemetry.REGISTRY).decode()
    assert "test_counter_total" in text
    assert 'kind="a"' in text


def test_gauge_and_histogram_registered():
    g = telemetry.gauge("test_gauge", "doc")
    g.set(7)
    h = telemetry.histogram("test_hist", "doc", ("q",), buckets=(0.1, 1.0, 10.0))
    h.labels(q="x").observe(0.5)
    text = generate_latest(telemetry.REGISTRY).decode()
    assert "test_gauge" in text
    assert "test_hist" in text


def test_get_tracer_returns_tracer():
    t = telemetry.get_tracer("unit-test")
    with t.start_as_current_span("span") as span:
        assert span is not None
