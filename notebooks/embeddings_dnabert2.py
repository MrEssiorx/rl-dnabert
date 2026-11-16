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
    # Эмбеддинги DNABERT-2

    Один исходный пик — один пример. Используем только chr1–22, chrX, chrY.
    Метка общей группы ставится по перекрытию **исходных** пиков не менее
    25% длины каждого пика; затем длинные пики обрезаются до 2560 bp вокруг summit.
    Эмбеддинг — среднее последнего скрытого слоя по непустым токенам.
    Инференс запускается отдельно кнопкой ниже. Сохранённые результаты
    загружаются при открытии notebook; оба графика используют одну проекцию PCA.
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
    ## Пакетный инференс

    Используется локальная копия checkpoint, закреплённая в `scripts/fetch_dnbert2.py`.
    Кнопка запускает весь подготовленный набор; для полного прогона нужен GPU.
    Результат: `artifacts/dnabert2_embeddings.pt` — эмбеддинги, метаданные
    каждого пика и revision checkpoint в одном файле.
    """)
    return


@app.cell
def _(mo):
    batch_size = mo.ui.number(start=1, step=1, value=16, label="Batch size")
    run_embeddings = mo.ui.run_button(label="Извлечь все эмбеддинги")
    mo.hstack([batch_size, run_embeddings])
    return batch_size, run_embeddings


@app.cell
def _(mo, run_embeddings, batch_size, dataset, examples):
    from rloops_llm_research.dnabert2_embeddings import (
        OUTPUT_PATH, extract_embeddings, load_embeddings, save_embeddings,
    )

    if run_embeddings.value:
        new_embeddings, peak_ids = extract_embeddings(dataset, batch_size=int(batch_size.value))
        save_embeddings(OUTPUT_PATH, new_embeddings, examples.drop(columns="sequence"), peak_ids)

    mo.stop(not OUTPUT_PATH.exists(),
            mo.md("Эмбеддинги ещё не сохранены. После кластерного прогона откройте notebook снова."))
    embeddings, metadata = load_embeddings()
    expected = examples.drop(columns="sequence").reset_index(drop=True)
    check_columns = ["peak_id", "source", "group", "chr", "window_start", "window_end"]
    if not metadata[check_columns].equals(expected[check_columns]):
        raise ValueError("Сохранённые метаданные не совпадают с текущей подготовкой пиков")
    mo.md(f"Загружено **{len(metadata):,} × {embeddings.shape[1]}** эмбеддингов.")
    return embeddings, metadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Два вида одной проекции

    PCA рассчитывается один раз по всем эмбеддингам. Оси одинаковы на обоих
    графиках; меняются только метки точек. Положение и видимое разделение
    в PCA сами по себе не доказывают существование биологических классов.
    """)
    return


@app.cell
def _(embeddings):
    import torch as _torch

    centered = embeddings.float() - embeddings.float().mean(dim=0)
    with _torch.random.fork_rng(devices=[]):
        _torch.manual_seed(2026)
        _, singular_values, directions = _torch.pca_lowrank(
            centered, q=8, center=False, niter=5
        )
    projection = (centered @ directions[:, :2]).numpy()
    explained = (singular_values[:2].square() / centered.square().sum()).tolist()
    return projection, explained


@app.cell
def _(mo):
    point_size = mo.ui.slider(start=1, stop=8, step=1, value=3, label="Размер точек")
    opacity = mo.ui.slider(start=0.1, stop=0.9, step=0.1, value=0.35, label="Прозрачность")
    mo.hstack([point_size, opacity])
    return point_size, opacity


@app.cell
def _(projection, explained, metadata, point_size, opacity):
    import matplotlib.pyplot as plt

    def draw_plot(column, categories, title):
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        colors = {"S9.6": "#247BA0", "dRNH": "#E76F51",
                  "S9.6_only": "#247BA0", "dRNH_only": "#E76F51",
                  "shared": "#7869AC"}
        if set(metadata[column]) != {value for value, _ in categories}:
            raise ValueError(f"Неожиданные метки в {column}: {set(metadata[column])}")
        for value, label in categories:
            selected = metadata[column].eq(value).to_numpy()
            ax.scatter(
                projection[selected, 0], projection[selected, 1],
                s=point_size.value, alpha=opacity.value,
                c=colors[value], edgecolors="none", rasterized=True,
                label=f"{label} (n={selected.sum():,})",
            )
        ax.set(title=title,
               xlabel=f"PC1 · {explained[0]:.1%} дисперсии",
               ylabel=f"PC2 · {explained[1]:.1%} дисперсии")
        ax.grid(color="#E6E9ED", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, markerscale=4, loc="best")
        return fig

    source_plot = draw_plot(
        "source", [("S9.6", "S9.6"), ("dRNH", "dRNH")],
        "DNABERT-2 · исходные наборы",
    )
    source_plot
    return draw_plot, plt


@app.cell
def _(draw_plot):
    groups_plot = draw_plot(
        "group",
        [("S9.6_only", "S9.6-only"), ("dRNH_only", "dRNH-only"),
         ("shared", "shared ≥25%")],
        "DNABERT-2 · группы по перекрытию",
    )
    groups_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Экспорт

    После сохранения эмбеддингов notebook можно открыть в `marimo edit` и
    менять оформление графиков. PDF с выполненными ячейками без листинга кода:

    ```bash
    .venv/bin/marimo export pdf notebooks/embeddings_dnabert2.py \
      -o artifacts/dnabert2_embeddings_report.pdf --no-include-inputs --force
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
