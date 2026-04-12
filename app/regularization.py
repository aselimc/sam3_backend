"""Orthogonal (rectilinear) polygon regularization for binary segmentation masks.

The goal is to take an irregular mask predicted by a segmentation model and
produce a "cleaner" mask whose boundary is a rectilinear polygon aligned with
the object's dominant orientation. This is the same kind of post-processing
typically applied to building footprints extracted from aerial imagery.

Pipeline (per connected component):
    1. Trace the external contour.
    2. Estimate the dominant edge direction via the minimum-area rotated rect.
    3. Rotate the contour so that direction is axis-aligned.
    4. Simplify the rotated contour with Douglas-Peucker.
    5. Snap each edge to be either horizontal or vertical, merging runs of
       same-orientation edges. The result is a strictly rectilinear polygon.
    6. Rotate back to image coordinates and rasterize.

The implementation only depends on numpy and OpenCV (cv2), both of which are
already available in this environment.
"""

from __future__ import annotations

import cv2
import numpy as np

# Components smaller than this many pixels are dropped entirely.
_MIN_COMPONENT_AREA = 16

# Douglas-Peucker tolerance, expressed as a fraction of the contour perimeter.
_SIMPLIFY_EPS_FRAC = 0.01

# Absolute lower bound on the simplification tolerance, in pixels.
_SIMPLIFY_EPS_MIN = 1.5


def regularize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a rectilinearized version of a binary mask.

    Args:
        mask: 2D array. Any non-zero pixel is treated as foreground. The
            output preserves the input dtype and uses the same foreground
            value as the input (255 for uint8 masks, 1 otherwise).

    Returns:
        A 2D array with the same shape and dtype as ``mask``, in which each
        connected component has been replaced by an axis-aligned-in-its-own-
        frame orthogonal polygon. If no usable component is found, an empty
        mask is returned.
    """
    if mask.ndim != 2:
        raise ValueError(f"regularize_mask expects a 2D array, got shape {mask.shape}")

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(mask)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask)

    out = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) < _MIN_COMPONENT_AREA:
            continue
        polygon = _regularize_contour(contour)
        if polygon is None or len(polygon) < 3:
            continue
        cv2.fillPoly(out, [polygon.astype(np.int32)], color=1)

    fg_value = 255 if mask.dtype == np.uint8 and mask.max() > 1 else 1
    return (out * fg_value).astype(mask.dtype)


def _regularize_contour(contour: np.ndarray) -> np.ndarray | None:
    """Regularize a single OpenCV contour into a rectilinear polygon."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    if w < 1 or h < 1:
        return None

    # cv2.minAreaRect returns angle in [-90, 0). Normalize so that rotating by
    # ``-angle`` aligns the rectangle's longer side with the x-axis.
    if w < h:
        angle += 90.0
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0

    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot_pts = _apply_affine(rot, pts)

    perimeter = cv2.arcLength(rot_pts.reshape(-1, 1, 2).astype(np.float32), True)
    eps = max(_SIMPLIFY_EPS_MIN, _SIMPLIFY_EPS_FRAC * perimeter)
    simplified = cv2.approxPolyDP(
        rot_pts.reshape(-1, 1, 2).astype(np.float32), eps, True
    ).reshape(-1, 2)

    rectilinear = _rectilinearize(simplified)
    if rectilinear is None:
        # Fall back to the rotated bounding rectangle, which is by construction
        # already a valid orthogonal polygon.
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    inv = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    return _apply_affine(inv, rectilinear)


def _apply_affine(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to an (N, 2) array of points."""
    homog = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (matrix @ homog.T).T.astype(np.float32)


def _rectilinearize(poly: np.ndarray) -> np.ndarray | None:
    """Snap a polygon's edges to be strictly horizontal or vertical.

    The input is assumed to already be expressed in a frame where the dominant
    edge direction is the x-axis. Each edge is classified as horizontal or
    vertical based on which delta is larger; consecutive same-orientation
    edges are merged into a single run. The walk produces a closed rectilinear
    polygon.

    Returns ``None`` if the polygon collapses (fewer than 4 distinct corners).
    """
    n = len(poly)
    if n < 4:
        return None

    orients = np.empty(n, dtype=np.int8)
    for i in range(n):
        dx = poly[(i + 1) % n, 0] - poly[i, 0]
        dy = poly[(i + 1) % n, 1] - poly[i, 1]
        orients[i] = 0 if abs(dx) >= abs(dy) else 1  # 0 = H, 1 = V

    # Find a starting edge whose orientation differs from its predecessor so
    # the cyclic walk doesn't split a run across the seam.
    start = -1
    for i in range(n):
        if orients[i] != orients[(i - 1) % n]:
            start = i
            break
    if start == -1:
        # All edges share the same orientation: degenerate.
        return None

    verts: list[tuple[float, float]] = []
    cx = float(poly[start, 0])
    cy = float(poly[start, 1])
    verts.append((cx, cy))

    cur_orient = int(orients[start])
    cur_target = poly[(start + 1) % n]

    for k in range(1, n):
        i = (start + k) % n
        if int(orients[i]) == cur_orient:
            cur_target = poly[(i + 1) % n]
            continue
        # Commit the run that just ended.
        if cur_orient == 0:
            cx = float(cur_target[0])
        else:
            cy = float(cur_target[1])
        verts.append((cx, cy))
        cur_orient = int(orients[i])
        cur_target = poly[(i + 1) % n]

    # Commit the final run.
    if cur_orient == 0:
        cx = float(cur_target[0])
    else:
        cy = float(cur_target[1])
    verts.append((cx, cy))

    # The polygon is closed, so the last vertex should coincide with the first.
    if (
        len(verts) >= 2
        and abs(verts[-1][0] - verts[0][0]) < 1e-3
        and abs(verts[-1][1] - verts[0][1]) < 1e-3
    ):
        verts.pop()

    if len(verts) < 4:
        return None

    return np.array(verts, dtype=np.float32)
