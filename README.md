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

You need Python 3.10 or newer. The recommended workflow uses
[uv](https://github.com/astral-sh/uv), which knows about per-package indexes
and so handles the PyTorch CUDA wheel selection automatically.

### Recommended: uv

```bash
# 1. Install uv (one-time, ~10 MB)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the repo root: create .venv and install everything
cd /path/to/scgg
uv sync                  # or `uv pip install -e .` to install into a pre-existing venv
source .venv/bin/activate
```

`pyproject.toml` already pins torch to the **CUDA 12.4 wheel index**
(`[tool.uv.sources]` block). If your cluster has a different CUDA, change the
URL in that section — e.g. `https://download.pytorch.org/whl/cu121` for CUDA
12.1 or `https://download.pytorch.org/whl/cpu` for CPU-only. The available
PyTorch CUDA builds are: cu121, cu124, cu126, cu128, cpu.

### Fallback: pip + install.sh

If you cannot use uv, the included `install.sh` detects your CUDA driver via
`nvidia-smi` and installs the matching PyTorch wheel, then `scgg`:

```bash
bash install.sh                      # auto-detect
bash install.sh --cuda 12.4          # force a specific wheel
bash install.sh --cpu                # CPU only
```

### Fallback: pure pip

```bash
# Pick ONE of these depending on your CUDA driver:
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only

# Then:
pip install -e /path/to/scgg
```

If `pip install -e .` fails with
`build backend is missing the 'build_editable' hook`, upgrade build tooling
first: `pip install --upgrade pip setuptools wheel`.

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
