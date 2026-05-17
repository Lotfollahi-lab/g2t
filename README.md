# scgg

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/sebastianbirk/scgg/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scgg

A very interesting piece of code

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need Python 3.10 or newer.

### Quick install (recommended)

```bash
bash install.sh
```

Detects your CUDA driver via `nvidia-smi`, installs the matching PyTorch
wheel, then installs `scgg` in editable mode. Pass `--cpu` for CPU-only,
`--cuda 12.4` to force a specific wheel, or `--skip-torch` if you've already
installed torch yourself.

### Manual install

### 1. Install PyTorch matching your CUDA driver

`scgg` declares `torch>=2.0` but does NOT pin a CUDA build, because the correct
wheel depends on your driver. Install it BEFORE `scgg` so pip doesn't pull the
default (latest-CUDA) wheel:

```bash
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA driver 550+ (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# NVIDIA driver 530+ (CUDA 12.1) — works with driver 12.4 too
pip install torch --index-url https://download.pytorch.org/whl/cu121

# NVIDIA driver 560+ (CUDA 12.6)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Check your driver's max CUDA version with `nvidia-smi` (top-right). Pick the
PyTorch wheel for a CUDA version ≤ that. If you see
`The NVIDIA driver on your system is too old (found version 1240X)` when
importing torch, you installed a wheel built for a newer CUDA than your
driver supports — uninstall and reinstall from the lower index above.

### 2. Install scgg

```bash
pip install -e /path/to/scgg
```

If `pip install -e .` fails with
`build backend is missing the 'build_editable' hook`, upgrade your build
tooling first:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/sebastianbirk/scgg/issues
[tests]: https://github.com/sebastianbirk/scgg/actions/workflows/test.yaml
[documentation]: https://scgg.readthedocs.io
[changelog]: https://scgg.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scgg.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scgg
