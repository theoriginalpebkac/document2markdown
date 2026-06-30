# Installing `doc2md.py`

Copy-paste setup for **macOS** (with and without Homebrew) and **Ubuntu Linux**.
For *what* each dependency is and *why*, see the
[Requirements section of the README](README.md#requirements); this file is just
the step-by-step.

`doc2md.py` needs three things:

1. **Python 3.11 or newer** — a hard requirement of `pdfmux`, the PDF engine.
2. **Python packages** from `requirements.txt` (installed into a virtualenv).
3. **Two system tools** that are *not* pip-installable:
   - **pandoc** — converts Word / HTML / EPUB / RTF / ODT.
   - **pdftotext** (poppler) — used for the PDF fidelity score. **Optional:** PDFs
     still convert without it; you just lose the similarity number.

> **Why a virtualenv?** Modern macOS (Homebrew Python) and Ubuntu refuse a bare
> `pip install` with an `externally-managed-environment` error, and `sudo pip`
> is a footgun. A per-project venv sidesteps both and is trivial to delete.

Pick **one** of the four paths below, then jump to
[Verify your install](#5-verify-your-install).

- [1. uv (fastest, all platforms)](#1-uv-fastest-all-platforms)
- [2. macOS — Homebrew](#2-macos--homebrew)
- [3. macOS — no Homebrew](#3-macos--no-homebrew)
- [4. Ubuntu Linux](#4-ubuntu-linux)

---

## 1. uv (fastest, all platforms)

[`uv`](https://docs.astral.sh/uv/) installs the *correct Python version for you*
and resolves the packages in one step — it sidesteps both the "Python too old"
and "externally-managed-environment" traps. It does **not** install the native
tools, so you still grab `pandoc`/`poppler` from your OS package manager.

```bash
# Install uv (macOS + Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Native tools (skip poppler if you don't need fidelity scores)
#   macOS:  brew install pandoc poppler
#   Ubuntu: sudo apt-get install -y pandoc poppler-utils

# Get the code and set up an isolated env on Python 3.12
git clone https://github.com/theoriginalpebkac/document2markdown.git
cd document2markdown
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then [verify](#5-verify-your-install).

---

## 2. macOS — Homebrew (recommended)

The most common macOS path, especially in enterprise environments.
[Homebrew](https://brew.sh/) installs Python *and* both native tools.
`python@3.14` is Homebrew's current latest (its default `python` formula); it's
well above pdfmux's 3.11 minimum.

```bash
# Python 3.14 (Homebrew's latest) + native tools
brew install python@3.14 pandoc poppler

# Get the code
git clone https://github.com/theoriginalpebkac/document2markdown.git
cd document2markdown

# Isolated env + Python packages
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If `pip install` can't find a wheel for a dependency (the ML stack
> occasionally lags the newest Python by a release), rebuild the venv on the
> previous version — `brew install python@3.13` and use `python3.13` — then
> re-run the install. See [Troubleshooting](#troubleshooting).

Then [verify](#5-verify-your-install).

---

## 3. macOS — no Homebrew

Use the official installers.

**Python** — download the macOS installer for **3.14** (the current release)
from [python.org/downloads](https://www.python.org/downloads/macos/) and run it.
It installs a `python3.14` command. (macOS does not ship a usable Python by
default.)

**pandoc** — download the macOS `.pkg` from the
[pandoc releases page](https://github.com/jgm/pandoc/releases/latest) and run it.

**pdftotext (poppler)** — there is **no official installer**, and this is the one
genuinely awkward dependency without a package manager. Your options:

- **Skip it.** It's optional — PDFs still convert, you just don't get a
  similarity score. Recommended unless you specifically need fidelity validation.
- Install via [MacPorts](https://www.macports.org/): `sudo port install poppler`
- Install via Conda/Miniconda: `conda install -c conda-forge poppler`

Then set up the project:

```bash
git clone https://github.com/theoriginalpebkac/document2markdown.git
cd document2markdown

python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then [verify](#5-verify-your-install).

---

## 4. Ubuntu Linux

> **Heads-up on the Python version.** Ubuntu **24.04** ships Python 3.12 — good.
> Ubuntu **22.04** ships Python **3.10**, which is **too old** for pdfmux. On
> 22.04 (or older) use the deadsnakes PPA path below. Check yours with
> `lsb_release -d` and `python3 --version`.

**Ubuntu 24.04+ (Python 3.12, the simple case):**

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip pandoc poppler-utils

git clone https://github.com/theoriginalpebkac/document2markdown.git
cd document2markdown

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Ubuntu 22.04 / 20.04 (need a newer Python via deadsnakes):**

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv pandoc poppler-utils

git clone https://github.com/theoriginalpebkac/document2markdown.git
cd document2markdown

python3.11 -m venv .venv          # pip comes bundled inside the venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then [verify](#5-verify-your-install).

---

## 5. Verify your install

With the venv **activated** (`source .venv/bin/activate`):

```bash
# Python is new enough? Must be 3.11 or higher.
python --version

# The CLI loads and its dependencies import cleanly.
python doc2md.py --help
```

`doc2md.py` also checks for `pandoc` and `pdftotext` when it runs a real
conversion — it prints install hints for anything missing rather than failing
silently, and missing tools degrade gracefully (affected files are recorded as
errors; the batch continues). To exercise the whole pipeline, point it at a
document:

```bash
python doc2md.py path/to/some.pdf
```

---

## Returning later

The venv persists. In a new shell, just reactivate it from the project folder:

```bash
cd document2markdown
source .venv/bin/activate
python doc2md.py ...
```

To upgrade after a `git pull`, with the venv active:

```bash
pip install -r requirements.txt --upgrade
```

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `error: externally-managed-environment` on `pip install` | You're outside a venv. Activate it first: `source .venv/bin/activate`. Never `sudo pip`. |
| `pdfmux` install fails / `requires Python >= 3.11` | Your venv was built with an old Python. Check `python --version`; rebuild the venv with `python3.14` (macOS) or the deadsnakes `python3.11` (Ubuntu 22.04). |
| `pip` can't find a wheel / build fails on the newest Python | The ML stack (torch via Docling) can lag the latest Python by a release. Rebuild the venv one version down — `brew install python@3.13`, then `python3.13 -m venv .venv` — and re-run `pip install -r requirements.txt`. |
| `pandoc: command not found` during conversion | Install the native tool for your OS (see your path above). Only affects Word/HTML/EPUB/RTF/ODT inputs. |
| `pdftotext` warning at startup | Optional tool not installed — PDFs still convert, you just don't get a fidelity score. Install poppler if you want it. |
| `command not found: python3.12` (macOS, no brew) | The python.org installer adds `python3.12`; open a fresh terminal so `PATH` updates, or use the full path it printed during install. |
