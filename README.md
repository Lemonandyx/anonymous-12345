# HarmoRAG

HarmoRAG is a query-adaptive RAG serving system with progressive retrieval
and cross-stage continuous batching. This repository contains the current
implementation, native scanner, benchmark entry points, tests, and plotting
code used by the accompanying paper.

## Requirements

- Linux x86-64
- Python 3.10
- CUDA-compatible NVIDIA GPU
- Faiss 1.8.0
- vLLM 0.11.2

## Installation

```bash
git clone https://github.com/Lemonandyx/anonymous-12345.git
cd anonymous-12345
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[runtime,figures,test]'
```

Build the optional native scanner after Faiss is installed:

```bash
FAISS_LIBRARY=faiss_avx2 ./build_native_scanner.sh
```

Use `FAISS_LIBRARY=faiss` when the installation provides `libfaiss.so`
instead of the AVX2 library.

## Data preparation

HarmoRAG expects an IVF-Flat index, its matching JSONL corpus, QA files, an
E5-large-v2 query encoder, and a generator model. QA records should contain
`question` and `golden_answers`; corpus records should contain `contents`.
E5 queries must use the `query: ` prefix.

Copy `configs/paper_nq_llama3_8b.yaml` and replace the placeholder paths with
local paths. Models, indexes, corpora, raw traces, and paper result tables are
not included in this repository.

## Usage

The compact Python interface is:

```python
from harmorag import Config, Dataset, ProgressiveAsyncPipeline

config = Config(config_file_path="configs/paper_nq_llama3_8b.yaml")
dataset = Dataset(config=config, dataset_path="/path/to/nq/test.jsonl")
result = ProgressiveAsyncPipeline(config).run(dataset)
result.save("output/nq_predictions.json")
```

For controlled serving experiments, use:

```bash
python benchmarks/benchmark_harmorag.py --help
```

The benchmark accepts the model, index, corpus, dataset, arrival-rate,
`nprobe`, and top-k settings as command-line arguments.

## Plotting

Paper plotting programs are retained under `figures/`. Provide your own
aggregated result tables. Scripts with command-line inputs document them via
`--help`; the remaining scripts expect matching tables below `data/`.
Generated plots are written under `output/`.

## Verification

```bash
python -m compileall -q harmorag benchmarks figures
pytest -q
```

## License

See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
