#!/usr/bin/env python3
"""Module that performs same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform a same convolution on multiple grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded_images = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    convolved = np.zeros((m, h, w))

    for row in range(h):
        for column in range(w):
            section = padded_images[
                :,
                row:row + kh,
                column:column + kw
            ]
            convolved[:, row, column] = np.sum(
                section * kernel,
                axis=(1, 2)
            )

    return convolved
