import marimo

__generated_with = "0.25.1"
app = marimo.App(layout_file="layouts/comparison.grid.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    matplotlib.style.use('default')
    return np, pd, plt


@app.cell
def _(pd):
    data_s96 = pd.read_csv("data/rloop_S96.narrowPeak", sep='\t', header=None, names=["chr", "start", "end", "name", "score", "strand", "signalValue", "pValue", "qValue", "peak"])
    data_s96["length"] = data_s96["end"] - data_s96["start"]
    return (data_s96,)


@app.cell
def _(pd):
    data_drnh = pd.read_csv("data/rlregions_dRNH.narrowPeak", sep='\t', header=None, names=["chr", "start", "end", "name", "score", "strand", "signalValue", "pValue", "qValue", "peak"])
    data_drnh["length"] = data_drnh["end"] - data_drnh["start"]
    return (data_drnh,)


@app.cell
def _(data_drnh, data_s96, plt):
    plt.figure(figsize=(10, 6))
    # xs = range(0, 3000)
    # ys_s96 = [(data_s96['score'] < x).sum() / len(data_s96) for x in xs]
    # ys_drnh = [(data_drnh['score'] < x).sum() / len(data_drnh) for x in xs]
    # plt.plot(xs, ys_s96, label='s9.6')
    # plt.plot(xs, ys_drnh, label='dRNH')
    plt.ecdf(data_s96["score"], label="s9.6")
    plt.ecdf(data_drnh["score"], label="dRNH")
    plt.legend()
    plt.title("score, ecdf")
    plt.ylim(0, 1)
    plt.xlim(0, 1500)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Эта картинка выглядет немного пугающе -- похоже, что некоторый загадочный `score` у dRNH ощутимо хуже чем у S9.6? Действительно ли это говорит о худшем качестве этой сборки?
    """)
    return


@app.cell
def _(data_s96, np):
    print("Корреляция score в S9.3 с lenght, length**0.5, length**0.4")
    print(np.corrcoef((data_s96["score"], data_s96["length"], data_s96["length"] ** 0.5), data_s96["length"] ** 0.4)[0][1:])
    return


@app.cell
def _(data_drnh, np):
    print("Корреляция score в dRNH с lenght, length**0.5, length**0.4")
    print(np.corrcoef((data_drnh["score"], data_drnh["length"], data_drnh["length"] ** 0.5), data_drnh["length"] ** 0.4)[0][1:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Полученный прежде факт: участки, находимые S9.6 в среднем **значительно** длиннее, чем находимые dRNH. Логичной гипотезой является тогда что длина участка и его `score` коррелируют, что подтверждается!
    """)
    return


@app.cell
def _(data_drnh, data_s96, plt):
    plt.figure(figsize=(10, 6))
    plt.ecdf(data_s96["length"], label="s9.6")
    plt.ecdf(data_drnh["length"], label="dRNH")
    plt.legend()
    plt.title("length (logscale), ecdf")
    plt.xscale('log')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пересечение пиков S9.6 и dRNH

    Координаты `.narrowPeak` — полуоткрытые интервалы `[start, end)`.
    Пересечение означает хотя бы 1 общий bp. Отдельно считаются все контиги
    и основные хромосомы chr1–22, chrX, chrY.
    """)
    return


@app.cell
def _(data_drnh, data_s96, pd):
    def grouped_intervals(frame):
        groups = {}
        for chrom, rows in frame.groupby("chr"):
            intervals = sorted(zip(rows["start"], rows["end"]))
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError(f"Overlapping peaks within one dataset on {chrom}")
            groups[chrom] = intervals
        return groups

    s96_intervals = grouped_intervals(data_s96)
    drnh_intervals = grouped_intervals(data_drnh)

    def summarize(chroms, label):
        n_s96 = sum(len(s96_intervals.get(chrom, ())) for chrom in chroms)
        n_drnh = sum(len(drnh_intervals.get(chrom, ())) for chrom in chroms)
        bases_s96 = sum(end - start for chrom in chroms for start, end in s96_intervals.get(chrom, ()))
        bases_drnh = sum(end - start for chrom in chroms for start, end in drnh_intervals.get(chrom, ()))
        hits_s96 = hits_drnh = pairs = shared_bases = 0
        for chrom in chroms:
            a, b = s96_intervals.get(chrom, ()), drnh_intervals.get(chrom, ())
            i = j = 0
            seen_a, seen_b = set(), set()
            while i < len(a) and j < len(b):
                shared = min(a[i][1], b[j][1]) - max(a[i][0], b[j][0])
                if shared > 0:
                    pairs += 1
                    shared_bases += shared
                    seen_a.add(i)
                    seen_b.add(j)
                if a[i][1] <= b[j][1]:
                    i += 1
                else:
                    j += 1
            hits_s96 += len(seen_a)
            hits_drnh += len(seen_b)
        union_bases = bases_s96 + bases_drnh - shared_bases
        return dict(group=label, s96_peaks=n_s96, drnh_peaks=n_drnh,
                    s96_with_overlap=hits_s96, drnh_with_overlap=hits_drnh,
                    overlapping_pairs=pairs, shared_bp=shared_bases,
                    union_bp=union_bases, jaccard_bp=shared_bases / union_bases)

    all_chroms = set(s96_intervals) | set(drnh_intervals)
    primary_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    overlap_summary = pd.DataFrame([
        summarize(all_chroms, "all contigs"),
        summarize(primary_chroms, "chr1-22, chrX, chrY"),
    ])
    return drnh_intervals, overlap_summary, s96_intervals


@app.cell
def _(overlap_summary):
    overlap_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Насколько перекрыт каждый пик

    Только chr1–22, chrX, chrY. Для каждого пика складываем все основания,
    общие с пиками другого набора, и делим на **длину этого пика**.
    Пик считается один раз на каждом пороге; процент в таблице — доля от
    **всех пиков своего набора** на выбранных хромосомах.
    """)
    return


@app.cell
def _(drnh_intervals, overlap_summary, pd, s96_intervals):
    primary_chroms_for_thresholds = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
    s96_coverages = []
    drnh_coverages = []
    for chrom in primary_chroms_for_thresholds:
        a, b = s96_intervals.get(chrom, ()), drnh_intervals.get(chrom, ())
        a_overlap = [0] * len(a)
        b_overlap = [0] * len(b)
        i = j = 0
        while i < len(a) and j < len(b):
            shared = min(a[i][1], b[j][1]) - max(a[i][0], b[j][0])
            if shared > 0:
                a_overlap[i] += shared
                b_overlap[j] += shared
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1
        s96_coverages.extend((covered, end - start) for (start, end), covered in zip(a, a_overlap))
        drnh_coverages.extend((covered, end - start) for (start, end), covered in zip(b, b_overlap))

    primary_summary = overlap_summary.loc[overlap_summary["group"] == "chr1-22, chrX, chrY"].iloc[0]
    assert all(0 <= covered <= length for covered, length in s96_coverages + drnh_coverages)
    assert sum(covered for covered, _ in s96_coverages) == primary_summary["shared_bp"]
    assert sum(covered for covered, _ in drnh_coverages) == primary_summary["shared_bp"]
    assert sum(covered > 0 for covered, _ in s96_coverages) == primary_summary["s96_with_overlap"]
    assert sum(covered > 0 for covered, _ in drnh_coverages) == primary_summary["drnh_with_overlap"]

    threshold_rows = []
    for name, coverages in (("S9.6", s96_coverages), ("dRNH", drnh_coverages)):
        for threshold in (10, 25, 50):
            count = sum(100 * covered >= threshold * length for covered, length in coverages)
            threshold_rows.append({
                "набор": name,
                "перекрыто не менее": f"{threshold}% длины пика",
                "число пиков": count,
                "всего пиков": len(coverages),
                "доля набора, %": round(100 * count / len(coverages), 2),
            })
    overlap_thresholds = pd.DataFrame(threshold_rows)
    return drnh_coverages, overlap_thresholds, s96_coverages


@app.cell
def _(overlap_thresholds):
    overlap_thresholds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Три группы для графика эмбеддингов

    Пик входит в общую группу, если с другим набором перекрыто **не менее 25%
    его собственной длины**. Поэтому два пересекающихся пика могут получить
    разные метки: порог проверяется отдельно для каждого пика.
    """)
    return


@app.cell
def _(drnh_coverages, pd, s96_coverages):
    s96_shared = sum(100 * covered >= 25 * length for covered, length in s96_coverages)
    drnh_shared = sum(100 * covered >= 25 * length for covered, length in drnh_coverages)
    embedding_groups = pd.DataFrame([
        {"группа": "только S9.6", "пиков S9.6": len(s96_coverages) - s96_shared,
         "пиков dRNH": 0, "всего пиков": len(s96_coverages) - s96_shared},
        {"группа": "только dRNH", "пиков S9.6": 0,
         "пиков dRNH": len(drnh_coverages) - drnh_shared,
         "всего пиков": len(drnh_coverages) - drnh_shared},
        {"группа": "общие", "пиков S9.6": s96_shared,
         "пиков dRNH": drnh_shared, "всего пиков": s96_shared + drnh_shared},
    ])
    assert embedding_groups["всего пиков"].sum() == len(s96_coverages) + len(drnh_coverages)
    return (embedding_groups,)


@app.cell
def _(embedding_groups):
    embedding_groups
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Геномные аннотации и внешние варианты

    Таблица ниже запускает `scripts/interval_overlap.py` для исходных пиков.
    В каждой строке показана доля **пиков** с перекрытием аннотации хотя бы на
    1 bp; знаменатель — пики на chr1–22, chrX, chrY. Один пик может пересечь
    несколько категорий. pELS и dELS — кандидатные энхансерные элементы ENCODE
    SCREEN Registry V4 (hg38). Z-DNA и GQ — опубликованные предсказания
    Z-DNABERT и GQ-DNABERT, переведённые из hg19 в hg38 через UCSC liftOver.
    Внешние наборы GWAS, GWAS ∩ QTL и OMIM появятся
    здесь после подготовки локальных BED-файлов по источникам Z-DNABERT.
    """)
    return


@app.cell
def _(mo, pd):
    import io
    import subprocess
    import sys
    from pathlib import Path

    annotation_dir = Path("data/annotations/gencode_v32")
    annotation_paths = {
        name: annotation_dir / f"{name}.bed"
        for name in ("promoter", "5utr", "exon", "intron", "3utr", "downstream", "distal_intergenic")
    }
    annotation_paths.update({
        "pELS": Path("data/annotations/encode_v4_pELS.bed"),
        "dELS": Path("data/annotations/encode_v4_dELS.bed"),
        "Z-DNA": Path("data/annotations/flipons/z_dnaberthg38.primary.bed"),
        "GQ": Path("data/annotations/flipons/gq_dnaberthg38.primary.bed"),
    })
    annotation_paths.update({
        "GWAS": Path("data/annotations/gwas.bed"),
        "GWAS ∩ eQTL": Path("data/annotations/gwas_eqtl.bed"),
        "GWAS ∩ sQTL": Path("data/annotations/gwas_sqtl.bed"),
        "GWAS ∩ edQTL": Path("data/annotations/gwas_edqtl.bed"),
        "OMIM": Path("data/annotations/omim.bed"),
    })
    available = {name: path for name, path in annotation_paths.items() if path.is_file()}
    missing = [name for name in annotation_paths if name not in available]
    if available:
        command = [sys.executable, "scripts/interval_overlap.py",
                   "--query", "S9.6=data/rlregions_S96.narrowPeak",
                   "--query", "dRNH=data/rlregions_dRNH.narrowPeak"]
        for _annotation_name, _annotation_path in available.items():
            command.extend(("--annotation", f"{_annotation_name}={_annotation_path}"))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        annotation_overlap = pd.read_csv(io.StringIO(result.stdout))
    else:
        annotation_overlap = pd.DataFrame()
    _labels = {
        "promoter": "Promoter", "5utr": "5′UTR", "exon": "Exon",
        "intron": "Intron", "3utr": "3′UTR", "downstream": "Downstream",
        "distal_intergenic": "Distal intergenic",
        "pELS": "Proximal enhancer-like (pELS)",
        "dELS": "Distal enhancer-like (dELS)", "GWAS": "GWAS",
        "Z-DNA": "Z-DNABERT: Z-DNA", "GQ": "GQ-DNABERT: GQ",
        "GWAS ∩ eQTL": "GWAS ∩ eQTL", "GWAS ∩ sQTL": "GWAS ∩ sQTL",
        "GWAS ∩ edQTL": "GWAS ∩ edQTL", "OMIM": "OMIM",
    }
    _lines = ["| Категория | S9.6 | dRNH |", "| --- | ---: | ---: |"]
    for _key in annotation_paths:
        _subset = annotation_overlap.loc[annotation_overlap["annotation"] == _key] if not annotation_overlap.empty else pd.DataFrame()
        if len(_subset) == 2:
            _values = dict(zip(_subset["query"], _subset["percent"]))
            _lines.append(f"| {_labels[_key]} | {_values['S9.6']:.2f}% | {_values['dRNH']:.2f}% |")
    _note = ("\n\nПока нет локальных данных: " + ", ".join(_labels[_key] for _key in missing) + ".") if missing else ""
    mo.md("\n".join(_lines) + _note)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
