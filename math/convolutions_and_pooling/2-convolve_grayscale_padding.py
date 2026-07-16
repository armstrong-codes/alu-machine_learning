#!/usr/bin/env python3
"""Module that performs convolution with custom padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Perform convolution on grayscale images with custom padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    convolved = np.zeros((m, output_h, output_w))

    for row in range(output_h):
        for column in range(output_w):
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
