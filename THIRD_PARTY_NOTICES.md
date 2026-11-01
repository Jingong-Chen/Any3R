# Third-party notices

Any3R is built on the following work. Their licenses apply to the corresponding parts of this repository and of the released weights.

## VGGT-Omega

Any3R's network trunk, camera head and dense head are those of VGGT-Omega, and the Any3R weights are fine-tuned from the VGGT-Omega weights. The code in `any3r/_vggt_omega/` is a copy of the VGGT-Omega inference package (imports renamed, otherwise unchanged).

Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the FAIR Noncommercial Research License, a copy of which is provided in [`LICENSE`](LICENSE). Under that license this repository, the Any3R weights and any derivative of them are distributed under the same terms, for noncommercial research use only.

## PanoVGGT

The panoramic position embedding and parts of the ray-field geometry (`any3r/geometry.py`, `any3r/aggregator.py`, `any3r/model.py`) are adapted from PanoVGGT, released under the MIT License:

```
MIT License

Copyright (c) 2026 Yijing Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Example data

The images in `examples/` are resampled from public datasets and keep their original licenses (each folder has a `SOURCE.md`):

| Folder | Source | License |
|---|---|---|
| `examples/street_mixed`, `examples/panocity_erp` | [PanoCity](https://huggingface.co/datasets/YijingGuo/PanoCity) | CC BY 4.0 |
| `examples/fiord_meetingroom` | [FIORD](https://zenodo.org/records/15181976) | CC BY 4.0 |
