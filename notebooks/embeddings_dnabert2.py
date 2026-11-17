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

    return (
        NARROWPEAK_COLUMNS,
        SequenceDataset,
        attach_fasta,
        crop_around_summit,
        mo,
        pd,
        prepare_peak_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Эмбеддинги DNABERT-2

    Один исходный пик — один пример. Используем только chr1–22, chrX, chrY.
    Метка общей группы ставится по перекрытию **исходных** пиков не менее
    25% длины каждого пика; затем длинные пики обрезаются до 2560 bp вокруг summit.
    Эмбеддинг — среднее последнего скрытого слоя по непустым токенам.
    Инференс запускается отдельно кнопкой ниже. Сохранённые результаты
    загружаются при открытии notebook; все графики используют одну PCA.
    Запускать ноутбук из `repo/`, чтобы пути `data/` работали.
    """)
    return


@app.cell
def _(NARROWPEAK_COLUMNS, pd):
    s96_raw = pd.read_csv(
        "data/rlregions_S96.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS
    )
    drnh_raw = pd.read_csv(
        "data/rlregions_dRNH.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS
    )
    return drnh_raw, s96_raw


@app.cell
def _(drnh_raw, prepare_peak_table, s96_raw):
    peaks = prepare_peak_table(s96_raw, drnh_raw, shared_threshold=0.25)
    return (peaks,)


@app.cell
def _(pd, peaks):
    group_counts = pd.crosstab(peaks["group"], peaks["source"])
    group_counts
    return


@app.cell
def _(crop_around_summit, peaks):
    windows = crop_around_summit(peaks, max_bp=2560)
    return (windows,)


@app.cell
def _(attach_fasta, windows):
    examples = attach_fasta(windows, "data/hg38.fa")
    assert examples["sequence"].str.len().eq(examples["window_length"]).all()
    return (examples,)


@app.cell
def _(examples):
    lengths = examples.groupby("source")["window_length"].agg(["count", "min", "median", "max"])
    return (lengths,)


@app.cell
def _(lengths):
    lengths
    return


@app.cell
def _(SequenceDataset, examples):
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
def _(batch_size, dataset, examples, mo, run_embeddings):
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

    PCA рассчитывается один раз по всем эмбеддингам. Выберите пару из первых
    трёх компонент; оба графика используют одни оси, меняются только метки
    точек. Знак каждой оси произволен. Видимое разделение в PCA само по себе
    не доказывает существование биологических классов.
    """)
    return


@app.cell
def _(embeddings):
    import torch as _torch

    mean_embedding = embeddings.float().mean(dim=0)
    centered = embeddings.float() - mean_embedding
    with _torch.random.fork_rng(devices=[]):
        _torch.manual_seed(2026)
        _, singular_values, directions = _torch.pca_lowrank(
            centered, q=8, center=False, niter=5
        )
    projection = (centered @ directions[:, :3]).numpy()
    explained = (singular_values[:3].square() / centered.square().sum()).tolist()
    return directions, explained, mean_embedding, projection


@app.cell
def _(mo):
    component_pair = mo.ui.dropdown(
        options=["PC1 × PC2", "PC1 × PC3", "PC2 × PC3"],
        value="PC1 × PC2", label="Оси PCA",
    )
    point_size = mo.ui.slider(start=1, stop=8, step=1, value=3, label="Размер точек")
    opacity = mo.ui.slider(start=0.1, stop=0.9, step=0.1, value=0.35, label="Прозрачность")
    mo.hstack([component_pair, point_size, opacity])
    return component_pair, opacity, point_size


@app.cell
def _(component_pair, explained, metadata, opacity, point_size, projection):
    import matplotlib.pyplot as plt

    def draw_plot(column, categories, title, pair=None):
        horizontal, vertical = {
            "PC1 × PC2": (0, 1), "PC1 × PC3": (0, 2), "PC2 × PC3": (1, 2),
        }[pair or component_pair.value]
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        colors = {"S9.6": "#247BA0", "dRNH": "#E76F51",
                  "S9.6_only": "#247BA0", "dRNH_only": "#E76F51",
                  "shared": "#7869AC"}
        if set(metadata[column]) != {value for value, _ in categories}:
            raise ValueError(f"Неожиданные метки в {column}: {set(metadata[column])}")
        for value, label in categories:
            selected = metadata[column].eq(value).to_numpy()
            ax.scatter(
                projection[selected, horizontal], projection[selected, vertical],
                s=point_size.value, alpha=opacity.value,
                c=colors[value], edgecolors="none", rasterized=True,
                label=f"{label} (n={selected.sum():,})",
            )
        ink = "#20252B"
        ax.set_title(title, color=ink)
        ax.set_xlabel(f"PC{horizontal + 1} · {explained[horizontal]:.1%} дисперсии", color=ink)
        ax.set_ylabel(f"PC{vertical + 1} · {explained[vertical]:.1%} дисперсии", color=ink)
        ax.tick_params(colors=ink)
        ax.grid(color="#E6E9ED", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(ink)
        ax.legend(frameon=False, markerscale=4, loc="best", labelcolor=ink)
        return fig

    source_plot = draw_plot(
        "source", [("S9.6", "S9.6"), ("dRNH", "dRNH")],
        "DNABERT-2 · исходные наборы",
    )
    source_plot
    return (draw_plot,)


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
    ## Дополнительные плоскости PCA

    Те же эмбеддинги S9.6 и dRNH в плоскостях PC1 × PC3 и PC2 × PC3.
    Оси взяты из той же PCA, что и графики выше. Смена плоскости меняет
    вид облаков, но сама по себе не доказывает разделение классов.
    """)
    return


@app.cell
def _(draw_plot):
    pc13_plot = draw_plot(
        "source", [("S9.6", "S9.6"), ("dRNH", "dRNH")],
        "DNABERT-2 · PC1 × PC3", pair="PC1 × PC3",
    )
    pc13_plot
    return


@app.cell
def _(draw_plot):
    pc23_plot = draw_plot(
        "source", [("S9.6", "S9.6"), ("dRNH", "dRNH")],
        "DNABERT-2 · PC2 × PC3", pair="PC2 × PC3",
    )
    pc23_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пики и случайные геномные негативы

    Негативы — 71 056 неперекрывающихся окон hg38 по 2 kb, выбранных после
    исключения окон с пиками S9.6/dRNH, ENCODE blacklist и >50% `N`.
    Первый график сохраняет PCA по одним пикам; второй заново рассчитывает
    PCA по пикам и негативам. Оба показывают только PC1 × PC2.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path as _NegativePath
    from rloops_llm_research.dnabert2_embeddings import load_embeddings as _load_negative_embeddings

    negative_path = _NegativePath("artifacts/dnabert2_negative_embeddings.pt")
    mo.stop(not negative_path.exists(), mo.md(
        "Эмбеддинги негативов ещё не сохранены; графики появятся после кластерного запуска."
    ))
    negative_embeddings, negative_metadata = _load_negative_embeddings(negative_path)
    if len(negative_metadata) != 71056 or not negative_metadata["source"].eq("negative").all():
        raise ValueError("Unexpected negative embeddings metadata")
    mo.md(f"Загружено **{len(negative_metadata):,}** негативов.")
    return negative_embeddings, negative_metadata


@app.cell
def _(embeddings, explained, mean_embedding, directions, metadata,
          negative_embeddings, negative_metadata, projection):
    import matplotlib.pyplot as _plt

    def draw_with_negatives(positive_xy, negative_xy, variances, title):
        fig, ax = _plt.subplots(figsize=(9, 7), layout="constrained")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.scatter(negative_xy[:, 0], negative_xy[:, 1], s=3, alpha=0.18,
                   c="#777777", edgecolors="none", rasterized=True,
                   label=f"Негативы (n={len(negative_metadata):,})")
        for source, color in (("S9.6", "#247BA0"), ("dRNH", "#E76F51")):
            selected = metadata["source"].eq(source).to_numpy()
            ax.scatter(positive_xy[selected, 0], positive_xy[selected, 1],
                       s=3, alpha=0.3, c=color, edgecolors="none", rasterized=True,
                       label=f"{source} (n={selected.sum():,})")
        ax.set_title(title)
        ax.set_xlabel(f"PC1 · {variances[0]:.1%} дисперсии")
        ax.set_ylabel(f"PC2 · {variances[1]:.1%} дисперсии")
        ax.grid(color="#E6E9ED", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, markerscale=4)
        return fig

    negative_on_peak_pca = (
        (negative_embeddings.float() - mean_embedding) @ directions[:, :2]
    ).numpy()
    peak_pca_with_negatives = draw_with_negatives(
        projection[:, :2], negative_on_peak_pca, explained,
        "DNABERT-2 · негативы на PCA по пикам",
    )
    peak_pca_with_negatives
    return (draw_with_negatives,)


@app.cell
def _(draw_with_negatives, embeddings, negative_embeddings):
    import torch as _combined_torch

    combined = _combined_torch.cat((embeddings.float(), negative_embeddings.float()))
    centered_combined = combined - combined.mean(dim=0)
    with _combined_torch.random.fork_rng(devices=[]):
        _combined_torch.manual_seed(2026)
        _, combined_singular_values, combined_directions = _combined_torch.pca_lowrank(
            centered_combined, q=8, center=False, niter=5
        )
    combined_xy = (centered_combined @ combined_directions[:, :2]).numpy()
    combined_explained = (
        combined_singular_values[:2].square() / centered_combined.square().sum()
    ).tolist()
    combined_pca_plot = draw_with_negatives(
        combined_xy[:len(embeddings)], combined_xy[len(embeddings):],
        combined_explained, "DNABERT-2 · PCA по пикам и негативам",
    )
    combined_pca_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Геномные категории по четырём группам

    Считаем **исходные пики**, а не обрезанные окна. Для каждой группы показана
    доля её пиков с пересечением аннотации хотя бы на 1 bp. Семь геномных
    категорий взяты из GENCODE v32, pELS (proximal enhancer-like) и dELS
    (distal enhancer-like) — из ENCODE SCREEN V4 (hg38).
    Категории могут перекрываться, поэтому проценты по столбцу не суммируются
    до 100%. Порог shared — ≥25% длины самого пика.
    """)
    return


@app.cell
def _(metadata, mo):
    from collections import defaultdict
    from html import escape
    from pathlib import Path
    import sys as _sys

    _sys.path.insert(0, str(Path("scripts").resolve()))
    from interval_overlap import count_hits, merge_intervals, read_intervals

    selections = {
        "S9.6 shared": metadata[(metadata["source"] == "S9.6") & (metadata["group"] == "shared")],
        "dRNH shared": metadata[(metadata["source"] == "dRNH") & (metadata["group"] == "shared")],
        "S9.6 only": metadata[metadata["group"] == "S9.6_only"],
        "dRNH only": metadata[metadata["group"] == "dRNH_only"],
    }
    queries = {}
    for label, table in selections.items():
        intervals = defaultdict(list)
        for chrom, start, end in table[["chr", "start", "end"]].itertuples(index=False, name=None):
            intervals[chrom].append((int(start), int(end)))
        queries[label] = intervals

    categories = {
        "promoter": "Promoter", "5utr": "5′UTR", "exon": "Exon",
        "intron": "Intron", "3utr": "3′UTR", "downstream": "Downstream",
        "distal_intergenic": "Distal intergenic",
        "pELS": "pELS", "dELS": "dELS",
    }
    annotation_paths = {
        key: Path("data/annotations/gencode_v32") / f"{key}.bed"
        for key in categories if key not in ("pELS", "dELS")
    }
    annotation_paths.update({
        "pELS": Path("data/annotations/encode_v4_pELS.bed"),
        "dELS": Path("data/annotations/encode_v4_dELS.bed"),
    })
    headers = list(selections)
    group_sizes = " · ".join(
        f"{escape(label)}: n={len(table):,}" for label, table in selections.items()
    )
    rows = []
    for key, label in categories.items():
        raw, _, _ = read_intervals(annotation_paths[key])
        annotation = {chrom: merge_intervals(rows) for chrom, rows in raw.items()}
        percentages = [
            f"{100 * count_hits(queries[name], annotation) / len(selections[name]):.2f}%"
            for name in selections
        ]
        rows.append((label, percentages))
    cell_style = "padding: 7px 10px; border-bottom: 1px solid #aaa5; white-space: nowrap;"
    heading = "<th style='" + cell_style + "text-align: left;'>Категория</th>" + "".join(
        f"<th style='{cell_style}text-align: right;'>{escape(header)}</th>" for header in headers
    )
    body = "".join(
        "<tr>" + f"<td style='{cell_style}'>{escape(label)}</td>" + "".join(
            f"<td style='{cell_style}text-align: right;'>{escape(value)}</td>" for value in percentages
        ) + "</tr>"
        for label, percentages in rows
    )
    mo.Html(
        "<div style='overflow-x: auto; max-width: 100%;'>"
        f"<p style='font-size: 0.8rem; margin: 0 0 8px;'>{group_sizes}</p>"
        f"<table style='border-collapse: collapse; width: 100%; font-size: 0.9rem;'>"
        f"<thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"
    )
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
