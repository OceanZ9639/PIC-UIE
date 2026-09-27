# PIC-UIE

The arXiv link will be added when it is available.

PIC-UIE is a lightweight underwater image enhancement model in YCbCr space.
Monocular depth is used only as training supervision. Inference takes RGB images and
does not require depth maps.

<video src="demo/videos/flsea_canyons_flatiron.mp4" controls></video>
<video src="demo/videos/flsea_canyons_u_canyon.mp4" controls></video>
<video src="demo/videos/flsea_red_sea_coral_table_loop.mp4" controls></video>
<video src="demo/videos/flsea_red_sea_dice_path.mp4" controls></video>

The released checkpoint is `weights/pic_uie.pth`.

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Run RGB-only inference with the released checkpoint:

```bash
python3 tools/test.py \
  --ckpt weights/pic_uie.pth \
  --in_dir /path/to/input_images \
  --out_dir results/pic_uie
```

Use `--size` to resize inputs before inference and `--device cpu` to run on CPU.

## Training

Training requires paired input/reference images and precomputed monocular depth maps.
The depth maps supervise the predicted transmission and are not passed to the model.

```bash
python3 tools/train.py \
  --in_dir /path/to/input_images \
  --gt_dir /path/to/reference_images \
  --depth_dir /path/to/depth_maps \
  --depth_invert \
  --size 256 --bs 16 --epochs 700 --lr 3e-4 \
  --c 16 --chroma lut --clut_size 9 --knots 32 --chroma_ds 1 \
  --depth_cond_chroma --tone curve --refine_y 16 --lum tgain \
  --ccm --colormatch --ema 0.999 --in_norm --global_stats \
  --w_lutsmooth 0.05 --w_lutmono 1.0 \
  --w_l1 1.0 --w_ssim 0.6 --w_msssim 0.4 --w_red 0.3 --w_depth 0.1 \
  --out_dir runs/pic_uie
```

Depth files can be grayscale PNG images or NumPy arrays and are matched by filename
stem. Use `--depth_invert` for disparity-like maps in which nearer regions have larger
values.
