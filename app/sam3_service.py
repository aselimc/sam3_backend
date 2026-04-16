import base64
import io
import os
import time

import numpy as np
import torch
from loguru import logger
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from app.metrics import (
    gpu_mem_allocated_bytes,
    gpu_mem_peak_bytes,
    inference_seconds,
)
from app.regularization import regularize_mask


class SAM3Service:
    def __init__(self, confidence_threshold: float = 0.5):
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)

    def _run_queries(
        self,
        image: Image.Image,
        queries: list[str],
        confidence_threshold: float,
        endpoint: str = "predict",
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run inference for each query, return list of (masks, boxes, scores)."""
        if confidence_threshold != self.processor.confidence_threshold:
            self.processor.set_confidence_threshold(confidence_threshold)

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for query in queries:
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(prompt=query, state=state)
                results.append((state["masks"], state["boxes"], state["scores"]))
        duration = time.perf_counter() - t0

        inference_seconds.labels(endpoint=endpoint).observe(duration)
        alloc = torch.cuda.memory_allocated() if cuda else 0
        peak = torch.cuda.max_memory_allocated() if cuda else 0
        gpu_mem_allocated_bytes.set(alloc)
        gpu_mem_peak_bytes.set(peak)
        logger.info(
            "inference done endpoint={} n_queries={} image_size={} "
            "duration_ms={:.1f} gpu_mem_alloc_mb={:.1f} gpu_mem_peak_mb={:.1f}",
            endpoint,
            len(queries),
            image.size,
            duration * 1000,
            alloc / (1024 * 1024),
            peak / (1024 * 1024),
        )
        return results

    @staticmethod
    def _resolve_regularize_flags(regularize: list[bool] | None, n_queries: int) -> list[bool]:
        if regularize is None:
            return [False] * n_queries
        if len(regularize) != n_queries:
            raise ValueError(
                "regularize must have the same length as queries "
                f"(got {len(regularize)} vs {n_queries})"
            )
        return list(regularize)

    @staticmethod
    def _mask_bbox(mask_u8: np.ndarray) -> list[float] | None:
        """Return [x1, y1, x2, y2] of foreground pixels, or None if empty."""
        ys, xs = np.where(mask_u8 > 0)
        if ys.size == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def _postprocess_mask(
        self,
        mask_u8: np.ndarray,
        original_box: list[float],
        regularize: bool,
    ) -> tuple[np.ndarray, list[float]]:
        """Apply optional regularization and return (mask, box)."""
        if not regularize:
            return mask_u8, original_box
        regularized = regularize_mask(mask_u8)
        new_box = self._mask_bbox(regularized)
        if new_box is None:
            # Regularization wiped the mask out — fall back to the original.
            return mask_u8, original_box
        return regularized, new_box

    def predict(
        self,
        image_path: str,
        queries: list[str],
        output_dir: str,
        confidence_threshold: float = 0.5,
        regularize: list[bool] | None = None,
    ) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        flags = self._resolve_regularize_flags(regularize, len(queries))
        raw_results = self._run_queries(image, queries, confidence_threshold, endpoint="predict")
        results = []
        for query, (masks, boxes, scores), do_reg in zip(queries, raw_results, flags):
            objects = []
            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                box = boxes[i].cpu().tolist()
                mask_np, box = self._postprocess_mask(mask_np, box, do_reg)

                mask_img = Image.fromarray(mask_np, mode="L")
                safe_query = query.replace(" ", "_").replace("/", "_")
                mask_filename = f"{image_name}_{safe_query}_{i}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                mask_img.save(mask_path)

                objects.append(
                    {
                        "mask_path": mask_path,
                        "box": box,
                        "score": scores[i].item(),
                    }
                )
            results.append({"query": query, "objects": objects})
        return results

    def predict_b64(
        self,
        image: Image.Image,
        queries: list[str],
        confidence_threshold: float = 0.5,
        regularize: list[bool] | None = None,
    ) -> list[dict]:
        flags = self._resolve_regularize_flags(regularize, len(queries))
        raw_results = self._run_queries(
            image, queries, confidence_threshold, endpoint="predict_upload"
        )
        results = []
        for query, (masks, boxes, scores), do_reg in zip(queries, raw_results, flags):
            objects = []
            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                box = boxes[i].cpu().tolist()
                mask_np, box = self._postprocess_mask(mask_np, box, do_reg)

                mask_img = Image.fromarray(mask_np, mode="L")
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                objects.append(
                    {
                        "mask_b64": mask_b64,
                        "box": box,
                        "score": scores[i].item(),
                    }
                )
            results.append({"query": query, "objects": objects})
        return results
