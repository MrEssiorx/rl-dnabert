import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from rloops_llm_research.peaks import NARROWPEAK_COLUMNS, prepare_peak_table
    from rloops_llm_research.cutting import crop_around_summit
    from rloops_llm_research.sequences import attach_fasta
    from rloops_llm_research.model_inputs import SequenceDataset

    return mo, pd, NARROWPEAK_COLUMNS, prepare_peak_table, crop_around_summit, attach_fasta, SequenceDataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Эмбеддинги DNABERT-2: подготовка входов

    Один исходный пик — один пример. Используем только chr1–22, chrX, chrY.
    Метка общей группы ставится по перекрытию **исходных** пиков не менее
    25% длины каждого пика; затем длинные пики обрезаются до 2560 bp вокруг summit.
    Код модели и получение эмбеддингов будут добавлены после отдельной проверки DNABERT-2.
    Запускать ноутбук из `repo/`, чтобы пути `data/` работали.
    """)
    return


@app.cell
def _(pd, NARROWPEAK_COLUMNS):
    s96_raw = pd.read_csv(
        "data/rlregions_S96.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS
    )
    drnh_raw = pd.read_csv(
        "data/rlregions_dRNH.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS
    )
    return s96_raw, drnh_raw


@app.cell
def _(s96_raw, drnh_raw, prepare_peak_table):
    peaks = prepare_peak_table(s96_raw, drnh_raw, shared_threshold=0.25)
    return (peaks,)


@app.cell
def _(peaks, pd):
    group_counts = pd.crosstab(peaks["group"], peaks["source"])
    group_counts
    return


@app.cell
def _(peaks, crop_around_summit):
    windows = crop_around_summit(peaks, max_bp=2560)
    return (windows,)


@app.cell
def _(windows, attach_fasta):
    examples = attach_fasta(windows, "data/hg38.fa")
    assert examples["sequence"].str.len().eq(examples["window_length"]).all()
    return (examples,)


@app.cell
def _(examples, pd):
    lengths = examples.groupby("source")["window_length"].agg(["count", "min", "median", "max"])
    preview = examples[[
        "peak_id", "chr", "start", "end", "summit", "window_start",
        "window_end", "source", "group", "overlap_fraction",
    ]].head(10).copy()
    preview["sequence_start"] = examples["sequence"].head(10).str[:30]
    pd.DataFrame(lengths)
    return lengths, preview


@app.cell
def _(lengths):
    lengths
    return


@app.cell
def _(preview):
    preview
    return


@app.cell
def _(examples, SequenceDataset):
    dataset = SequenceDataset(examples)
    assert len(dataset) == len(examples)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Далее

    После проверки checkpoint и способа извлечения одного вектора здесь появятся
    токенизатор, пакетный запуск DNABERT-2, сохранение эмбеддингов и два графика.
    До этого этапа таблица `examples` сохраняет и последовательности, и метаданные.
    """)
    return


if __name__ == "__main__":
    app.run()
