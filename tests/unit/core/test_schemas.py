import pytest
from pydantic import ValidationError

from packages.core.schemas import VersionedModel


def test_versioned_default():
    m = VersionedModel()
    assert m.version == "1"


def test_versioned_extra_forbidden():
    with pytest.raises(ValidationError):
        VersionedModel.model_validate({"version": "1", "rogue": True})


def test_versioned_only_v1():
    with pytest.raises(ValidationError):
        VersionedModel.model_validate({"version": "2"})
