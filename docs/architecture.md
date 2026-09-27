# PIC-UIE Architecture

## Overview

PIC-UIE separates luminance restoration from chroma correction. A compact encoder runs
on a fixed 256 x 256 representation and predicts transform parameters. These
parameters are then applied at the input resolution.

```text
RGB input
  -> fixed-resolution encoder
  -> luminance parameters: transmission, veil, gain, tone curve
  -> chroma parameters: CbCr lookup table and correction strength
  -> optional Y refinement and global YCbCr alignment
  -> RGB output
```

## Encoder

`TinyEncoder` contains three stride-2 convolution stages. Global average pooling
produces the vector used by image-level transform heads. With `global_stats` enabled,
raw RGB mean and standard deviation are projected into the same vector. Instance
normalization can be enabled independently with `in_norm`.

The released models use identity-oriented initialization for transform heads so that
training begins close to an unchanged image.

## Luminance branch

The luminance branch estimates a transmission map `t`, a veiling-light term, and a
bounded gain. It applies

```text
y1 = y - (1 - t) * veil
y2 = y1 * (1 + gain * (1 - t))
```

followed by a monotone one-dimensional tone curve. A small Y-only residual block
provides local refinement without directly changing chroma.

## Chroma branch

The chroma branch predicts an image-adaptive 2D lookup table over Cb and Cr. The table
is evaluated at reduced spatial resolution and upsampled to the input resolution. A
transmission-conditioned gate controls correction strength when
`depth_cond_chroma` is enabled.

## Global alignment

The released configuration enables a 3 x 3 affine transform in YCbCr space and an
optional covariance-matching transform. Both operate after the luminance and chroma
branches and before conversion back to RGB.

## Use of depth

Monocular depth is used only during training. The loss aligns `-log(t)` with relative,
distance-like depth using a scale- and shift-invariant correlation term. The encoder
always receives RGB, and inference does not load or estimate depth.
