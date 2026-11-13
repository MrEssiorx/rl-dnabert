# R-loop research code

Code for studying R-loop regions in the human hg38 reference with genomic language models. This repository is the reproducible code component of a developing research project. The experimental protocol and reproduction instructions will be completed as methods stabilize.

## Code map

| Path | Purpose |
| --- | --- |
| `notebooks/initial_analysis_s96.py` | marimo exploration of S9.6 consensus peaks |
| `notebooks/initial_analysis_drnh.py` | corresponding exploration of dRNH consensus peaks |
| `notebooks/comparison.py` | comparison of both peak sets |
| `notebooks/rlbase_checkout.py` | exploration of RLBase region table and provenance |
| `notebooks/embeddings_dnabert2.py` | peak/sequence preparation for the first frozen-embedding experiment; model inference is pending |
| `scripts/fetch_data.py` | checksum-verified retrieval of known public inputs and local table check |
| `scripts/fetch_enhancers.py` | separate checksum-verified download of SCREEN Registry V4 enhancer-like cCREs |
| `scripts/fetch_flipon_predictions.py` | separately fetch author Z-DNA/GQ predictions and lift hg19 coordinates to hg38 |
| `scripts/prepare_gencode_regions.py` | derive seven hg38 region BED files from a pinned GENCODE GTF |
| `scripts/interval_overlap.py` | count input intervals intersecting each local BED annotation |
| `scripts/fetch_dnbert2.py` | download the pinned DNABERT-2 checkpoint |
| `scripts/smoke_dnbert2.py` | check offline tokenization and embedding inference |
| `src/rloops_llm_research/peaks.py` | primary-chromosome filtering and per-peak S9.6/dRNH overlap labels |
| `src/rloops_llm_research/cutting.py` | replaceable summit-centered coordinate strategy |
| `src/rloops_llm_research/sequences.py` | indexed FASTA extraction while retaining peak metadata |
| `src/rloops_llm_research/model_inputs.py` | model-agnostic dataset and tokenizer batch adapter; no DNABERT-2 implementation yet |
| `data/` | local inputs; downloaded base model weights belong in `data/models/`; ignored by Git |
| `artifacts/` | generated logs, figures, metrics, checkpoints and fine-tuned weights; ignored by Git |

The notebooks are marimo Python files, not `.ipynb`. Install `uv` and run `uv sync --locked` first. From this directory, use `uv run marimo edit notebooks/<name>.py` or `uv run python notebooks/<name>.py`. If `uv` is not on `PATH` but a local `.venv` already exists, use `.venv/bin/uv`. The notebooks use paths relative to this directory. Python 3.14 and dependencies are specified in `.python-version`, `pyproject.toml` and `uv.lock`.

Run `python scripts/fetch_data.py` to reuse verified peak files or download missing ones. Add `--with-genome` to also get the UCSC hg38 FASTA; this needs about 4 GB in `data/` plus temporary disk space. Add `--verify-only` for a read-only check. The local `rlregions_table.rda` version does not yet have a confirmed public download URL; the script verifies its checksum when provided. See the script for exact URLs and hashes. A fresh clone has no data. `artifacts/` is created locally for outputs and is never committed.

Current notebooks are exploratory and do not yet reproduce a completed training experiment. Decisions about labeling, negatives, splits, model pooling and evaluation remain open. Keep model downloads under `data/models/<model>/<revision>/` and run output under `artifacts/<experiment>/`; log and checkpoint names can evolve with the experiment.

The first embedding notebook keeps peak metadata and DNA strings together in a pandas table. Preparation uses only chr1-22, chrX, chrY; labels `shared` from at least 25% overlap of each original peak; and crops longer peaks to 2560 bp within peak boundaries around the summit. It does not create a separate sequence export. Run `uv run python -m unittest discover -s tests -v` for synthetic coordinate, FASTA, and batch-metadata checks. DNABERT-2 loading, pooling and inference will be added after the model setup is verified.

## DNABERT-2 setup and smoke test

The first model is [`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M), pinned to commit `7bce263b15377fc15361f52cfab88f8b586abda0`. The checkpoint includes custom Python code; `trust_remote_code=True` loads that code from the downloaded local snapshot. Inspect a new revision before changing this pin. From `repo/`:

```bash
uv sync --locked
uv run python scripts/fetch_dnbert2.py
uv run python scripts/smoke_dnbert2.py --device cpu
```

The smoke test uses two short sequences and an attention-mask-weighted mean of the final token states, including the model's special tokens. This is a provisional single-vector extraction rule for the first embedding experiment. It checks a 768-dimensional finite output and reports token lengths, load/inference time, process peak RSS and (on GPU) peak allocated CUDA memory. The checkpoint lives under ignored `data/models/`; no model files are committed. The first experiment's 2560 bp sequence cap is a study choice, not a verified model limit.

For cHARISMa, clone this code repository on the login node, then run `git pull` to update it. Download the pinned checkpoint separately into `data/models/DNABERT-2-117M/<revision>/`; keep input data, credentials and SSH configuration outside Git. Create/sync the environment with the locked dependencies on the login node; verify the installed PyTorch CUDA runtime against the compute node's driver before a GPU run. Execute the smoke test on a compute node through Slurm, for example from the cluster clone:

```bash
git clone <repository-url> rloops-llm-research
cd rloops-llm-research
uv sync --locked
uv run python scripts/fetch_dnbert2.py
mkdir -p artifacts
sbatch -A <project-id> -p gpu-ef-quick -G 1 -c 4 --time=00:15:00 \
  --output=artifacts/dnbert2-smoke-%j.log \
  --wrap='cd "$SLURM_SUBMIT_DIR" && .venv/bin/python scripts/smoke_dnbert2.py --device cuda'
```

Check the resulting Slurm log and `sacct` for elapsed time and peak memory. The official [cHARISMa run instructions](https://hpc.hse.ru/instructions/run) allow installation and lightweight checks on the login node, while computations belong on compute nodes; `gpu-ef-quick` is a preemptible student-accessible A100/H100 queue. The cluster procedure is a prepared recipe until an actual GPU job verifies it.
On the tested cHARISMa configuration, Slurm rejected `--mem=16G` with “Memory specification can not be satisfied”; omit that option unless the cluster's memory accounting changes.

## Genomic annotation pilot

This analysis is independent of model training. Download the chromosome-only
[GENCODE v32 comprehensive GTF](https://www.gencodegenes.org/human/release_32.html)
from `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_32/gencode.v32.annotation.gtf.gz`
to `data/gencode.v32.annotation.gtf.gz` (SHA-256
`da9dc827e9a111c9bde60c55f55b7d37ffebc4649fc01f0745a670240e939ed4`).
The local reference is UCSC hg38, and `data/hg38.fa.fai` supplies chromosome
lengths. From `repo/`, run:

```bash
python scripts/prepare_gencode_regions.py data/gencode.v32.annotation.gtf.gz data/hg38.fa.fai data/annotations/gencode_v32
python scripts/fetch_enhancers.py
python scripts/fetch_flipon_predictions.py
python scripts/interval_overlap.py \
  --query S96=data/rlregions_S96.narrowPeak \
  --query dRNH=data/rlregions_dRNH.narrowPeak \
  --annotation promoter=data/annotations/gencode_v32/promoter.bed \
  --annotation 5utr=data/annotations/gencode_v32/5utr.bed \
  --annotation exon=data/annotations/gencode_v32/exon.bed \
  --annotation intron=data/annotations/gencode_v32/intron.bed \
  --annotation 3utr=data/annotations/gencode_v32/3utr.bed \
  --annotation downstream=data/annotations/gencode_v32/downstream.bed \
  --annotation distal_intergenic=data/annotations/gencode_v32/distal_intergenic.bed \
  --annotation pELS=data/annotations/encode_v4_pELS.bed \
  --annotation dELS=data/annotations/encode_v4_dELS.bed \
  --annotation Z-DNA=data/annotations/flipons/z_dnaberthg38.primary.bed \
  --annotation GQ=data/annotations/flipons/gq_dnaberthg38.primary.bed \
  --output artifacts/initial_region_overlaps.csv
```

The GTF is converted from 1-based closed to 0-based half-open coordinates.
Promoter is 3 kb upstream of each annotated transcript TSS; downstream is
300 bp after each transcript end. Exons and introns are built from all
transcripts, and 5′/3′ UTRs are assigned by strand relative to that
transcript's CDS. Distal intergenic is the complement of annotated gene
bodies plus the chosen promoter and downstream flanks. Transcripts can overlap,
so categories are independent and can overlap. The output percentage is the
number of input intervals with at least 1 bp of overlap divided by the number
on chr1–22, chrX, chrY. These rules approximate the Z-DNABERT categories;
they are not claimed to reproduce the paper's exact annotation pipeline.

The pELS and dELS files come from the [SCREEN Registry V4 GRCh38/hg38
downloads](https://screen.wenglab.org/downloads). They represent proximal
and distal *enhancer-like candidate cis-regulatory elements*, not a claim
that every element is an active enhancer in every tissue. The downloaded
files contain 249,464 pELS and 1,469,205 dELS. SHA-256 checksums are pinned
in `scripts/fetch_enhancers.py`. `--verify-only` checks local copies without
accessing the network. Only `interval_overlap.py` filters out noncanonical
chromosomes for the working analysis; the source BED files are kept intact.


`notebooks/comparison.py` runs the same overlap script and displays available
annotations. Additional GWAS, GWAS ∩ eQTL, GWAS ∩ sQTL, GWAS ∩ edQTL and OMIM
BED files will follow the sources and intersections reported in Z-DNABERT:
GWAS Catalog v1.0, GTEx v8 and OMIM Morbid. These files are not part of
`fetch_data.py` or cluster training setup.

The [Z-DNABERT Z-DNA BED](https://github.com/mitiau/Z-DNABERT/blob/main/beds/readme.md)
(`HG19_thr_0.25_minlen_12_pad.bed`, threshold 0.25, minimum length 12)
and [GQ-DNABERT GQ BED](https://github.com/mitiau/G-DNABERT/tree/master/beds)
(`HG19_thr_0.25_minlen_6.bed`, threshold 0.25, minimum length 6) are
published in hg19. `scripts/fetch_flipon_predictions.py` preserves both
original BEDs, downloads the UCSC `hg19ToHg38.over.chain.gz` and Linux x86_64
`liftOver`, and remaps with `-minMatch=0.95`. It writes both complete lifted
BEDs and `.primary.bed` files restricted to chr1–22, chrX, chrY. Source,
chain, binary, and derived output SHA-256 values are pinned in the script;
`--verify-only` checks local copies without downloading. These downloads
are separate from training-data setup. The same overlap script measures the
fraction of S9.6/dRNH intervals intersecting each predicted Z-DNA/GQ set.
