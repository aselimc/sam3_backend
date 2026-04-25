import pytest

from packages.core.cancel import CancelCheck, CancelRequested


def test_no_trip_no_raise():
    chk = CancelCheck()
    chk()  # noop


def test_trip_then_call_raises():
    chk = CancelCheck()
    assert chk.tripped is False
    chk.trip()
    assert chk.tripped is True
    with pytest.raises(CancelRequested):
        chk()
