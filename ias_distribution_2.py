# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "fastexcel==0.13.0",
#     "marimo",
#     "openpyxl==3.1.5",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "polars==1.26.0",
#     "pyarrow==19.0.1",
#     "xlsxwriter==3.2.2",
# ]
# ///

import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(
        r"""
        # Task: Create time series of various quantiles of inflation expectations (discrete treatment)

        The response variable is discrete and ordinal. Instead of using the built-in continuous quartile functions, we now compute the quantiles by finding the smallest discrete value for which the cumulative frequency reaches the desired threshold. The plotting also uses markers to better reflect the discrete nature of the values.
        """
    )
    return


@app.cell
def _():
    import csv
    import polars as pl
    import pandas as pd
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    return csv, go, mo, pd, pl, px


@app.cell
def _(pd, pl):
    def clean_ias(df: pl.DataFrame): 
        df = df.filter(pl.col("q2a_agg1").is_not_null())
        df = df.filter(pl.col("q2a_agg1") != "19") # Filter out don't knows.
        df = df.with_columns([
            pl.when(pl.col("age") == 8).then(1)
            .when(pl.col("age") == 7).then(6)
            .otherwise(pl.col("age"))
            .alias("age")
        ])
        return df

    def convert_yyyyqq_to_datetime(yyyyqq: str) -> pd.Timestamp:
        year = int(yyyyqq[:4])
        quarter = int(yyyyqq[4:])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
        return pd.Timestamp(year=year, month=month, day=1)
    return clean_ias, convert_yyyyqq_to_datetime


@app.cell
def _(clean_ias, convert_yyyyqq_to_datetime, pl):
    # ---------------------------
    # Step 1. Load Excel and clean
    # ---------------------------
    df = pl.read_excel(
        "../ias_analysis/data/individual-responses-xlsx.xlsx",
        sheet_name="Dataset",
        columns=["yyyyqq", "age", "weight", "q2a_agg1"]
    )
    df = clean_ias(df)

    df = df.with_columns([
        pl.col('yyyyqq')
        .map_elements(convert_yyyyqq_to_datetime)
        .alias('date')
    ])

    df = df.with_columns(pl.col("q2a_agg1").cast(pl.Float64))

    df.head()

    # ---------------------------
    # Step 2. Process with Polars to group
    # ---------------------------
    return (df,)


@app.cell
def _(df, pl):
    import math

    def compute_discrete_quantile(arr, q):
        """
        Compute the discrete quantile for an array of values.
        For a given q (e.g., 0.05), the quantile is defined as the smallest value
        such that at least q*100% of the data is less than or equal to that value.
        """
        sorted_arr = sorted(arr)
        n = len(sorted_arr)
        # Use ceiling to determine the index in a zero-indexed list.
        idx = math.ceil(q * n) - 1
        return sorted_arr[idx]

    # Compute discrete quantiles for each date group using the custom function.
    result = df.group_by("date").agg([
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.05)).alias("5th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.10)).alias("10th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.25)).alias("25th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.50)).alias("50th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.75)).alias("75th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.90)).alias("90th"),
        pl.col("q2a_agg1").map_elements(lambda x: compute_discrete_quantile(x, 0.95)).alias("95th"),
    ])

    result = result.sort(by='date')
    result_pd = result.to_pandas()
    result
    return compute_discrete_quantile, math, result, result_pd


@app.cell
def _(px, result_pd):
    readable_mapping = {
         1: "Down by 1% or less",
         2: "Down by 1% but less than 2%",
         3: "Down by 2% but less than 3%",
         4: "Down by 3% but less than 4%",
         5: "Down by 4% but less than 5%",
         6: "Down by 5% or more",
         7: "not changed",
         8: "up by 1% or less",
         9: "up by 1% but less than 2%",
         10: "up by 2% but less than 3%",
         11: "up by 3% but less than 4%",
         12: "up by 4% but less than 5%",
         13: "up by 5% but less than 6%",
         14: "up by 6% but less than 7%",
         15: "up by 7% but less than 8%",
         16: "up by 8% but less than 9%",
         17: "up by 9% but less than 10%",
         18: "up by 10% or more",
         # 19: "no idea",
         20: "up by 10% by less than 11%",
         21: "up by 11% by less than 12%",
         22: "up by 12% by less than 13%",
         23: "up by 13% by less than 14%",
         24: "up by 14% by less than 15%",
         25: "up by 15% or more"
    }

    tickval_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25]
    # Create the line chart using the numeric quantile values.
    fig = px.line(result_pd, x="date", y=['5th', '10th', '25th', '50th', '75th', '90th', '95th'],
                  title='Inflation Expectations by Quantiles (Discrete Treatment)',
                  markers=True)
    # Update the y-axis ticks to display human-readable descriptions.
    fig.update_yaxes(
         tickmode="array",
         tickvals=tickval_list,
         ticktext=[readable_mapping.get(i, str(i)) for i in tickval_list]
    )
    fig.show()

    # # Plot with markers to better represent the discrete quantile values.
    # fig = px.line(result_pd, x="date", y=['5th', '10th', '25th', '50th', '75th', '90th', '95th'],
    #               title='Inflation Expectations by Quantiles (Discrete Treatment)',
    #               markers=True)
    # fig.show()
    return fig, readable_mapping, tickval_list


@app.cell
def _(result_pd):
    output_excel = "inflation_attitudes_quantiles_discrete.xlsx"
    result_pd.to_excel(output_excel, index=False)
    return (output_excel,)


@app.cell
def _(go, result_pd):
    # Convert the ordinal values to z-scores for analysis
    quantiles = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"]
    quantile_data = result_pd[quantiles]
    quantile_zscores = (quantile_data - quantile_data.mean()) / quantile_data.std()

    # Add date back in for plotting
    quantile_zscores["date"] = result_pd["date"]

    # Melt the z-score dataframe for plotting
    melted_z = quantile_zscores.melt(id_vars="date", var_name="Quantile", value_name="Z-score")

    # Create the z-score line plot
    fig_zscore = go.Figure()

    for quantile in melted_z["Quantile"].unique():
        subset = melted_z[melted_z["Quantile"] == quantile]
        fig_zscore.add_trace(go.Scatter(x=subset["date"], y=subset["Z-score"], mode="lines", name=quantile))

    fig_zscore.update_layout(
        title="Standardized Inflation Expectations by Quantile",
        xaxis_title="Date",
        yaxis_title="Z-score",
        legend_title="Quantile",
        template="plotly_white"
    )

    fig_zscore.show()
    return (
        fig_zscore,
        melted_z,
        quantile,
        quantile_data,
        quantile_zscores,
        quantiles,
        subset,
    )


@app.cell
def _(go, result_pd):
    # Calculate Interquantile Range (IQR)
    result_pd["IQR_75_25"] = result_pd["75th"] - result_pd["25th"]
    result_pd["IQR_90_10"] = result_pd["90th"] - result_pd["10th"]

    # Create Interquantile Range chart
    fig_iqr = go.Figure()
    fig_iqr.add_trace(go.Scatter(x=result_pd["date"], y=result_pd["IQR_75_25"], mode="lines", name="75th - 25th"))
    fig_iqr.add_trace(go.Scatter(x=result_pd["date"], y=result_pd["IQR_90_10"], mode="lines", name="90th - 10th"))

    fig_iqr.update_layout(
        title="Interquantile Range of Inflation Expectations",
        xaxis_title="Date",
        yaxis_title="Ordinal Range (Categories)",
        legend_title="Range",
        template="plotly_white"
    )

    fig_iqr.show()
    return (fig_iqr,)


@app.cell
def _(quantile_zscores, result_pd):
    from pandas import ExcelWriter

    # Prepare z-scores with date for export
    zscore_export = quantile_zscores.copy()

    # Combine all into one Excel file
    output_path = "inflation_expectations_quantiles_analysis.xlsx"

    with ExcelWriter(output_path, engine="xlsxwriter") as writer:
        result_pd.to_excel(writer, index=False, sheet_name="Original Quantiles")
        zscore_export.to_excel(writer, index=False, sheet_name="Z-Scores")
        result_pd[["date", "IQR_75_25", "IQR_90_10"]].to_excel(writer, index=False, sheet_name="Interquantile Ranges")

    output_path
    return ExcelWriter, output_path, writer, zscore_export


if __name__ == "__main__":
    app.run()
