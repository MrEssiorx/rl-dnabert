import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    matplotlib.style.use('default')
    return np, pd, plt


@app.cell
def _(pd):
    raw_data = pd.read_csv("data/rlregions_dRNH.narrowPeak", sep='\t', header=None, names=["chr", "start", "end", "name", "score", "strand", "signalValue", "pValue", "qValue", "peak"])
    return (raw_data,)


@app.cell
def _(raw_data):
    raw_data.head(5)
    return


@app.cell
def _(raw_data):
    print((raw_data["signalValue"] == 0).all(), (raw_data['pValue'] == 0).all(), (raw_data['qValue'] == 0).all(), (raw_data['strand'] == '.').all())
    return


@app.cell
def _(raw_data):
    data = raw_data.drop(columns=['signalValue', 'pValue', 'qValue', 'strand'])
    return (data,)


@app.cell
def _(data):
    data.head(5)
    return


@app.cell
def _(data):
    len(data)
    return


@app.cell
def _(data, plt):
    plt.figure(figsize=(10, 6))
    data["length"] = data["end"] - data["start"]
    reference_length_bp = 512 * 5  # historical comparison point, not a model limit
    q =  data["length"].quantile(0.95)
    plt.hist(data["length"][data["length"] <= q], bins=20)
    plt.title("распределение длин пиков (самые короткие 95%)")
    plt.axvline(reference_length_bp, color="red", label="условный порог 2560 bp (не лимит модели)")
    plt.legend()
    plt.show()
    return (reference_length_bp,)


@app.cell
def _(data, reference_length_bp, np, plt):
    plt.figure(figsize=(10, 6))
    plt.hist(data["length"], bins=np.logspace(2, 5, 60))
    plt.xscale("log")
    plt.title("распределение длин пиков (logscale)")
    plt.axvline(reference_length_bp, color="red", label="условный порог 2560 bp (не лимит модели)")
    plt.legend()
    plt.show()
    return


@app.cell
def _(data, reference_length_bp):
    print("Доля пиков длиной не более 2560 bp (условный порог):", (data["length"] <= reference_length_bp).sum() / len(data["length"]))
    print("Число пиков длиной не более 2560 bp (условный порог):", (data["length"] <= reference_length_bp).sum())
    return


@app.cell
def _(data):
    (data["peak"] == -1).sum()
    return


@app.cell
def _(data):
    data["rel"] = data["peak"] / data["length"]
    return


@app.cell
def _(data, np, plt):
    plt.figure(figsize=(10, 6))
    plt.hist(data["rel"], bins=np.linspace(0, 1, 50), range=(0, 1))
    plt.title("расределение относительного положения summit на всем пике")
    plt.show()
    return


@app.cell
def _(data, plt):
    plt.figure(figsize=(10, 6))
    plt.ecdf(data["length"])
    plt.title("ecdf, logscale")
    plt.xscale("log")
    plt.show()
    return


@app.cell
def _(data):
    data["chr"].unique()
    return


@app.cell
def _(data):
    canon = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    canon_chrs_data = data[data["chr"].isin(canon)]

    print(f"Entries: {len(data)} -> {len(canon_chrs_data)}")
    print(f"bps: {data['length'].sum()} -> {canon_chrs_data['length'].sum()}")
    return canon, canon_chrs_data


@app.cell
def _(canon, canon_chrs_data, plt):
    plt.figure(figsize=(10, 6))
    plt.title("количественное распределение пиков по хромосомам")
    canon_chrs_data["chr"].value_counts().reindex(canon).plot.barh()
    return


@app.cell
def _(data, plt):
    plt.figure(figsize=(10, 6))
    plt.hist(data['score'], bins=20)
    plt.title('распределение score')
    plt.ylabel('n')
    plt.xlabel('score')
    return


if __name__ == "__main__":
    app.run()
