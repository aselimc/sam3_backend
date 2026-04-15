HTTP Endpoints
==============

The OpenAPI spec below is regenerated from the live FastAPI app on every docs
build (see ``_ext/gen_openapi.py``).

Usage examples
--------------

``POST /predict`` — path-based, masks saved to disk:

.. code-block:: bash

   curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{
              "image_path": "/abs/path/to/image.jpg",
              "queries": ["bee", "flower"],
              "output_dir": "data/outputs",
              "confidence_threshold": 0.4,
              "regularize": [false, true]
            }'

``POST /predict/upload`` — multipart upload, masks returned as base64:

.. code-block:: bash

   curl -X POST http://localhost:8000/predict/upload \
        -F "image=@assets/bee.jpg" \
        -F "queries=bee" \
        -F "queries=flower" \
        -F "confidence_threshold=0.4"

``GET /metrics`` — Prometheus exposition (enabled when ``SAM3_ENABLE_METRICS=true``).

OpenAPI reference
-----------------

.. openapi:: openapi.json
   :format: markdown
