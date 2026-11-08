from __future__ import annotations

import numpy as np


def _subiteration(curr_image, fg_pixels, first):
    pixels = list(fg_pixels)
    if not pixels:
        return [], []
    rows = np.fromiter((point[0] for point in pixels), dtype=np.int64)
    cols = np.fromiter((point[1] for point in pixels), dtype=np.int64)
    active = curr_image[rows, cols] == 1
    rows, cols = rows[active], cols[active]
    if rows.size == 0:
        return [], []

    p2 = curr_image[rows - 1, cols]
    p3 = curr_image[rows - 1, cols + 1]
    p4 = curr_image[rows, cols + 1]
    p5 = curr_image[rows + 1, cols + 1]
    p6 = curr_image[rows + 1, cols]
    p7 = curr_image[rows + 1, cols - 1]
    p8 = curr_image[rows, cols - 1]
    p9 = curr_image[rows - 1, cols - 1]
    neighbors = (
        (p2 > 0).astype(np.int8)
        + (p3 > 0)
        + (p4 > 0)
        + (p5 > 0)
        + (p6 > 0)
        + (p7 > 0)
        + (p8 > 0)
        + (p9 > 0)
    )
    transitions = (
        ((p2 == 0) & (p3 > 0)).astype(np.int8)
        + ((p3 == 0) & (p4 > 0))
        + ((p4 == 0) & (p5 > 0))
        + ((p5 == 0) & (p6 > 0))
        + ((p6 == 0) & (p7 > 0))
        + ((p7 == 0) & (p8 > 0))
        + ((p8 == 0) & (p9 > 0))
        + ((p9 == 0) & (p2 > 0))
    )
    if first:
        directional = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
    else:
        directional = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
    remove = (neighbors >= 2) & (neighbors <= 6) & directional & (transitions == 1)
    zero_pixels = list(zip(rows[remove].tolist(), cols[remove].tolist()))

    next_pixels = set()
    for row, col in zero_pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and curr_image[row + dr, col + dc] == 1:
                    next_pixels.add((row + dr, col + dc))
    return zero_pixels, list(next_pixels)


def first_subiteration(curr_image, fg_pixels):
    return _subiteration(curr_image, fg_pixels, True)


def second_subiteration(curr_image, fg_pixels):
    return _subiteration(curr_image, fg_pixels, False)
