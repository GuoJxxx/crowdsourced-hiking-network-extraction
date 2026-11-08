from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as nd

import subiterations


skeleton_images_path = "skeleton_images/"


class GrayscaleSkeleton:
    def skeletonize(self, image):
        image = nd.grey_closing(image, footprint=circle(8), mode="constant", cval=0.0)
        image = add_zero_mat(image)
        previous_binary = np.zeros_like(image)
        image_bit_depth = (image.dtype.itemsize * 8) // 2
        print(f"image_bit_depth: {image_bit_depth}")
        thresholds = [2**x for x in range(image_bit_depth, 3, -1)] + list(
            range(15, 0, -1)
        )
        print(f"image_thresholds: {thresholds}")
        Path(skeleton_images_path).mkdir(parents=True, exist_ok=True)

        for threshold in thresholds:
            print(f"curr_threshold: {threshold}")
            thresholded = np.where(image < threshold, 0, image)
            binary = (thresholded > 0).astype(np.int16)
            cv2.imwrite(
                str(Path(skeleton_images_path) / f"binary_{threshold}.png"),
                (binary * 255).astype(np.uint8),
            )
            current = self.thin_pixels(previous_binary + binary)
            cv2.imwrite(
                str(Path(skeleton_images_path) / f"skeleton_{threshold}.png"),
                current.astype(np.uint8),
            )
            print(f"curr_skeleton max: {current.max()}")
            previous_binary = current
        return remove_zero_mat(previous_binary)

    def thin_pixels(self, image):
        neighbors = nd.convolve(
            (image > 0).astype(np.int16),
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            mode="constant",
            cval=0.0,
        )
        foreground = np.where((image == 1) & (neighbors >= 2) & (neighbors <= 6))
        check_pixels = list(zip(foreground[0], foreground[1]))
        iteration = 0
        while check_pixels:
            iteration += 1
            image, first_check = self.apply_subiteration(
                subiterations.first_subiteration, image, check_pixels
            )
            image, second_check = self.apply_subiteration(
                subiterations.second_subiteration,
                image,
                list(set(check_pixels + first_check)),
            )
            check_pixels = list(set(first_check + second_check))
            if iteration % 10 == 0 or not check_pixels:
                print(
                    f"thin iteration {iteration}: next {len(check_pixels)}",
                    flush=True,
                )

        foreground = np.where(image == 1)
        image, _ = self.apply_subiteration(
            self.empty_pools,
            image,
            list(zip(foreground[0], foreground[1])),
        )
        return image

    @staticmethod
    def apply_subiteration(function, image, foreground):
        if not foreground:
            return image, []
        zero_pixels, next_pixels = function(image, foreground)
        if zero_pixels:
            rows, cols = zip(*zero_pixels)
            image[np.asarray(rows), np.asarray(cols)] = 0
        return image, list(next_pixels)

    @staticmethod
    def empty_pools(image, foreground):
        zero_pixels = []
        for row, col in foreground:
            window = image[row - 1 : row + 2, col - 1 : col + 2] > 0
            if int(window.sum()) - int(image[row, col] > 0) > 6:
                zero_pixels.append((row, col))
        return zero_pixels, []


def add_zero_mat(image):
    return np.pad(image, 1, mode="constant", constant_values=0)


def remove_zero_mat(image):
    return image[1:-1, 1:-1]


def circle(radius):
    x, y = np.mgrid[: (2 * radius) + 1, : (2 * radius) + 1]
    values = (x - radius) ** 2 + (y - radius) ** 2
    return (values <= radius**2).astype(np.uint8)


if __name__ == "__main__":
    input_filename = str(sys.argv[1])
    output_filename = str(sys.argv[2])
    print(f"input filename: {input_filename}")
    print(f"output filename: {output_filename}")
    input_kde = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)
    if input_kde is None:
        raise FileNotFoundError(input_filename)
    started = time.time()
    skeleton = GrayscaleSkeleton().skeletonize(input_kde)
    print(f"total elapsed time: {time.time() - started} seconds")
    cv2.imwrite(output_filename, skeleton.astype(np.uint8))

