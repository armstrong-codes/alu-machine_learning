#!/usr/bin/env python3
"""Module that performs convolution using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Perform convolution on images using multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((h - 1) * sh + kh - h) + 1) // 2
        pw = (((w - 1) * sw + kw - w) + 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1

    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            section = padded_images[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw,
                :
            ]

            for k in range(nc):
                convolved[:, i, j, k] = np.sum(
                    section * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
