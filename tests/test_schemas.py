"""Tests for request/response schema validation."""

import pytest
from app.schemas import PredictRequest


class TestPredictRequest:
    def test_valid_request(self):
        req = PredictRequest(
            image_path="/tmp/test.jpg",
            queries=["building", "road"],
            output_dir="/tmp/out",
        )
        assert req.confidence_threshold == 0.5
        assert req.regularize is None

    def test_regularize_length_mismatch(self):
        with pytest.raises(ValueError, match="regularize must have the same length"):
            PredictRequest(
                image_path="/tmp/test.jpg",
                queries=["building"],
                output_dir="/tmp/out",
                regularize=[True, False],
            )

    def test_regularize_matching_length(self):
        req = PredictRequest(
            image_path="/tmp/test.jpg",
            queries=["building", "road"],
            output_dir="/tmp/out",
            regularize=[True, False],
        )
        assert req.regularize == [True, False]
