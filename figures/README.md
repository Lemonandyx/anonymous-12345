# Plotting code

This directory contains the current plotting programs used by the HarmoRAG
paper. Experimental measurements are intentionally excluded from the public
repository. Supply your own aggregated CSV, JSON, or XLSX inputs and run the
corresponding script from the repository root.

```bash
python figures/figure08_e2e_latency/plot.py --help
python figures/figure13_active_overlap/plot.py --help
python figures/plot_retrieval_latency.py --help
```

Other figure-specific programs follow the same layout. Generated PDF, SVG,
PNG, and TIFF files should be written under `output/`, which is ignored by
Git. Scripts without explicit input flags expect their aggregate tables under
the matching `data/figureXX_*` directory.
