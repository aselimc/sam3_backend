import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Service:
    def __init__(self, confidence_threshold: float = 0.5):
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(
            self.model, confidence_threshold=confidence_threshold
        )

    def _run_queries(
        self,
        image: Image.Image,
        queries: list[str],
        confidence_threshold: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run inference for each query, return list of (masks, boxes, scores)."""
        if confidence_threshold != self.processor.confidence_threshold:
            self.processor.set_confidence_threshold(confidence_threshold)

        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for query in queries:
                state = self.processor.set_image(image)
                state = self.processor.set_text_prompt(prompt=query, state=state)
                results.append((state["masks"], state["boxes"], state["scores"]))
        return results

    def predict(
        self,
        image_path: str,
        queries: list[str],
        output_dir: str,
        confidence_threshold: float = 0.5,
    ) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        raw_results = self._run_queries(image, queries, confidence_threshold)
        results = []
        for query, (masks, boxes, scores) in zip(queries, raw_results):
            objects = []
            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_np, mode="L")
                safe_query = query.replace(" ", "_").replace("/", "_")
                mask_filename = f"{image_name}_{safe_query}_{i}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                mask_img.save(mask_path)

                objects.append({
                    "mask_path": mask_path,
                    "box": boxes[i].cpu().tolist(),
                    "score": scores[i].item(),
                })
            results.append({"query": query, "objects": objects})
        return results

    def predict_b64(
        self,
        image: Image.Image,
        queries: list[str],
        confidence_threshold: float = 0.5,
    ) -> list[dict]:
        raw_results = self._run_queries(image, queries, confidence_threshold)
        results = []
        for query, (masks, boxes, scores) in zip(queries, raw_results):
            objects = []
            for i in range(masks.shape[0]):
                mask_np = masks[i, 0].cpu().numpy().astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_np, mode="L")
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                objects.append({
                    "mask_b64": mask_b64,
                    "box": boxes[i].cpu().tolist(),
                    "score": scores[i].item(),
                })
            results.append({"query": query, "objects": objects})
        return results
