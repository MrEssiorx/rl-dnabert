import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyreadr
    import pandas as pd

    return mo, pd, pyreadr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkout `rlegions_table.rda`
    """)
    return


@app.cell
def _(pyreadr):
    result = pyreadr.read_r("data/rlregions_table.rda")
    print(result.keys())
    return (result,)


@app.cell
def _(result):
    table_name = list(result.keys())[0]
    df = result[table_name]
    df.head()
    return (df,)


@app.cell
def _(df):
    len(df)
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkout `.narrowPeak` files
    """)
    return


@app.cell
def _(pd):
    precise_peaks = pd.read_csv("data/rlregions_dRNH.narrowPeak", sep='\t', header=None)
    precise_peaks.head()
    return (precise_peaks,)


@app.cell
def _(precise_peaks):
    len(precise_peaks)
    return


@app.cell
def _(pd):
    other_peaks = pd.read_csv("data/rlregions_S96.narrowPeak", sep='\t', header=None)
    other_peaks.head()
    return (other_peaks,)


@app.cell
def _(other_peaks):
    len(other_peaks)
    return


if __name__ == "__main__":
    app.run()
