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

__generated_with = "0.12.6"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(
        r"""
        # Task: Create time series of various quantiles of inflation expectations (discrete treatment) - 1 year ahead

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
def _(go, math, pd, pl, px):
    def quantile_maker(question="q2a_agg1"):
        """Wrapper function"""

        # Define question-specific parameters
        question_config = {
            "q2a_agg1": {
                "no_idea_code": "19",
                "response_map": {
                    1.0: -0.5, 2.0: -1.5, 3.0: -2.5, 4.0: -3.5, 5.0: -4.5, 6.0: -5.5,
                    7.0: 0.0, 8.0: 0.5, 9.0: 1.5, 10.0: 2.5, 11.0: 3.5, 12.0: 4.5,
                    13.0: 5.5, 14.0: 6.5, 15.0: 7.5, 16.0: 8.5, 17.0: 9.5, 18.0: 10.5,
                    19.0: None, 20.0: 10.5, 21.0: 11.5, 22.0: 12.5, 23.0: 13.5, 24.0: 14.5, 25.0: 15.5
                },
                "response_labels": {
                    1.0: "Down by 1% or less", 2.0: "Down by 1% to <2%", 3.0: "Down by 2% to <3%",
                    4.0: "Down by 3% to <4%", 5.0: "Down by 4% to <5%", 6.0: "Down by 5% or more",
                    7.0: "No change", 8.0: "Up by 1% or less", 9.0: "Up by 1% to <2%", 10.0: "Up by 2% to <3%",
                    11.0: "Up by 3% to <4%", 12.0: "Up by 4% to <5%", 13.0: "Up by 5% to <6%",
                    14.0: "Up by 6% to <7%", 15.0: "Up by 7% to <8%", 16.0: "Up by 8% to <9%",
                    17.0: "Up by 9% to <10%", 18.0: "Up by 10% or more", 19.0: "No idea",
                    20.0: "Up by 10% to <11%", 21.0: "Up by 11% to <12%", 22.0: "Up by 12% to <13%",
                    23.0: "Up by 13% to <14%", 24.0: "Up by 14% to <15%", 25.0: "Up by 15% or more"
                }
            },
            "q2b_agg1": {
                "no_idea_code": "19",
                "response_map": {
                    1.0: -0.5, 2.0: -1.5, 3.0: -2.5, 4.0: -3.5, 5.0: -4.5, 6.0: -5.5,
                    7.0: 0.0, 8.0: 0.5, 9.0: 1.5, 10.0: 2.5, 11.0: 3.5, 12.0: 4.5,
                    13.0: 5.5, 14.0: 6.5, 15.0: 7.5, 16.0: 8.5, 17.0: 9.5, 18.0: 10.5,
                    19.0: None, 20.0: 10.5, 21.0: 11.5, 22.0: 12.5, 23.0: 13.5, 24.0: 14.5, 25.0: 15.5
                },
                "response_labels": {
                    1.0: "Down by 1% or less", 2.0: "Down by 1% to <2%", 3.0: "Down by 2% to <3%",
                    4.0: "Down by 3% to <4%", 5.0: "Down by 4% to <5%", 6.0: "Down by 5% or more",
                    7.0: "No change", 8.0: "Up by 1% or less", 9.0: "Up by 1% to <2%", 10.0: "Up by 2% to <3%",
                    11.0: "Up by 3% to <4%", 12.0: "Up by 4% to <5%", 13.0: "Up by 5% to <6%",
                    14.0: "Up by 6% to <7%", 15.0: "Up by 7% to <8%", 16.0: "Up by 8% to <9%",
                    17.0: "Up by 9% to <10%", 18.0: "Up by 10% or more", 19.0: "No idea",
                    20.0: "Up by 10% to <11%", 21.0: "Up by 11% to <12%", 22.0: "Up by 12% to <13%",
                    23.0: "Up by 13% to <14%", 24.0: "Up by 14% to <15%", 25.0: "Up by 15% or more"
                }
            },
            "q2c_agg1": {
                "no_idea_code": "19",
                "response_map": {
                    1.0: -0.5, 2.0: -1.5, 3.0: -2.5, 4.0: -3.5, 5.0: -4.5, 6.0: -5.5,
                    7.0: 0.0, 8.0: 0.5, 9.0: 1.5, 10.0: 2.5, 11.0: 3.5, 12.0: 4.5,
                    13.0: 5.5, 14.0: 6.5, 15.0: 7.5, 16.0: 8.5, 17.0: 9.5, 18.0: 10.5,
                    19.0: None, 20.0: 10.5, 21.0: 11.5, 22.0: 12.5, 23.0: 13.5, 24.0: 14.5, 25.0: 15.5
                },
                "response_labels": {
                    1.0: "Down by 1% or less", 2.0: "Down by 1% to <2%", 3.0: "Down by 2% to <3%",
                    4.0: "Down by 3% to <4%", 5.0: "Down by 4% to <5%", 6.0: "Down by 5% or more",
                    7.0: "No change", 8.0: "Up by 1% or less", 9.0: "Up by 1% to <2%", 10.0: "Up by 2% to <3%",
                    11.0: "Up by 3% to <4%", 12.0: "Up by 4% to <5%", 13.0: "Up by 5% to <6%",
                    14.0: "Up by 6% to <7%", 15.0: "Up by 7% to <8%", 16.0: "Up by 8% to <9%",
                    17.0: "Up by 9% to <10%", 18.0: "Up by 10% or more", 19.0: "No idea",
                    20.0: "Up by 10% to <11%", 21.0: "Up by 11% to <12%", 22.0: "Up by 12% to <13%",
                    23.0: "Up by 13% to <14%", 24.0: "Up by 14% to <15%", 25.0: "Up by 15% or more"
                }
            },
            "q1b": {
                "no_idea_code": "14",
                "response_map": {
                    1.0: -2.0, 2.0: 0.0, 3.0: 0.5, 4.0: 1.5, 5.0: 2.5, 6.0: 3.5, 7.0: 4.5,
                    8.0: 5.5, 9.0: 6.5, 10.0: 7.5, 11.0: 8.5, 12.0: 9.5, 13.0: 10.5,
                    14.0: None, 15.0: 10.5, 16.0: 11.5, 17.0: 12.5, 18.0: 13.5, 19.0: 14.5, 20.0: 15.5
                },
                "response_labels": {
                    1.0: "Gone down", 2.0: "No change", 3.0: "Up by 1% or less", 
                    4.0: "Up by 1% to <2%", 5.0: "Up by 2% to <3%", 6.0: "Up by 3% to <4%",
                    7.0: "Up by 4% to <5%", 8.0: "Up by 5% to <6%", 9.0: "Up by 6% to <7%",
                    10.0: "Up by 7% to <8%", 11.0: "Up by 8% to <9%", 12.0: "Up by 9% to <10%",
                    13.0: "Up by 10% or more", 14.0: "No idea", 15.0: "Up by 10% to <11%",
                    16.0: "Up by 11% to <12%", 17.0: "Up by 12% to <13%", 18.0: "Up by 13% to <14%",
                    19.0: "Up by 14% to <15%", 20.0: "Up by 15% or more"
                }
            },
            "q1c": {
                "no_idea_code": "19",
                "response_map": {
                    1.0: -0.5, 2.0: -1.5, 3.0: -2.5, 4.0: -3.5, 5.0: -4.5, 6.0: -5.5,
                    7.0: 0.0, 8.0: 0.5, 9.0: 1.5, 10.0: 2.5, 11.0: 3.5, 12.0: 4.5,
                    13.0: 5.5, 14.0: 6.5, 15.0: 7.5, 16.0: 8.5, 17.0: 9.5, 18.0: 10.5,
                    19.0: None, 20.0: 10.5, 21.0: 11.5, 22.0: 12.5, 23.0: 13.5, 24.0: 14.5, 25.0: 15.5
                },
                "response_labels": {
                    1.0: "Down by 1% or less", 2.0: "Down by 1% to <2%", 3.0: "Down by 2% to <3%",
                    4.0: "Down by 3% to <4%", 5.0: "Down by 4% to <5%", 6.0: "Down by 5% or more",
                    7.0: "No change", 8.0: "Up by 1% or less", 9.0: "Up by 1% to <2%", 10.0: "Up by 2% to <3%",
                    11.0: "Up by 3% to <4%", 12.0: "Up by 4% to <5%", 13.0: "Up by 5% to <6%",
                    14.0: "Up by 6% to <7%", 15.0: "Up by 7% to <8%", 16.0: "Up by 8% to <9%",
                    17.0: "Up by 9% to <10%", 18.0: "Up by 10% or more", 19.0: "No idea",
                    20.0: "Up by 10% to <11%", 21.0: "Up by 11% to <12%", 22.0: "Up by 12% to <13%",
                    23.0: "Up by 13% to <14%", 24.0: "Up by 14% to <15%", 25.0: "Up by 15% or more"
                }
            }
        }

        config = question_config[question]
        response_map_f64 = config["response_map"]
        response_labels = config["response_labels"]
        no_idea_code = config["no_idea_code"]

        def clean_ias(df: pl.DataFrame):
            df = df.filter(pl.col(question).is_not_null())
            df = df.filter(pl.col(question) != no_idea_code)  # Filter out don't knows.
            df = df.with_columns(
                [
                    pl.when(pl.col("age") == 8)
                    .then(1)
                    .when(pl.col("age") == 7)
                    .then(6)
                    .otherwise(pl.col("age"))
                    .alias("age")
                ]
            )
            return df

        def convert_yyyyqq_to_datetime(yyyyqq: str) -> pd.Timestamp:
            year = int(yyyyqq[:4])
            quarter = int(yyyyqq[4:])
            month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter]
            return pd.Timestamp(year=year, month=month, day=1)

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

        ### LOAD THE DATA
        df = pl.read_excel(
            "./data/individual-responses.xlsx",
            sheet_name="Dataset",
            columns=["yyyyqq", "age", "weight", question],
        )
        df = clean_ias(df)

        df = df.with_columns(
            [
                pl.col("yyyyqq")
                .map_elements(convert_yyyyqq_to_datetime)
                .alias("date")
            ]
        )
        df = df.with_columns(pl.col(question).cast(pl.Float64))

        df = df.with_columns(
            pl.col(question).replace(response_map_f64).alias(question)
        ).filter(
            pl.col(question).is_not_null()
        )

        # Compute discrete quantiles for each date group using the custom function.
        result = df.group_by("date").agg(
            [
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.05))
                .alias("5th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.10))
                .alias("10th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.25))
                .alias("25th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.50))
                .alias("50th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.75))
                .alias("75th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.90))
                .alias("90th"),
                pl.col(question)
                .map_elements(lambda x: compute_discrete_quantile(x, 0.95))
                .alias("95th"),
            ]
        )

        result = result.sort(by="date")
        result_pd = result.to_pandas()

        # Create a sorted list of tickvals and labels based on interpolated values
        sorted_items = sorted(
            ((code, val) for code, val in response_map_f64.items() if val is not None),
            key=lambda x: x[1]
        )

        tickvals = [val for _, val in sorted_items]
        ticktext = [response_labels[code] for code, _ in sorted_items]

        # Create the line chart using the numeric quantile values.
        question_titles = {
            "q2a_agg1": "Inflation Expectations Quantiles (1 Year Ahead)",
            "q2b_agg1": "Inflation Expectations Quantiles (2 Years Ahead)", 
            "q2c_agg1": "Inflation Expectations Quantiles (5 Years Ahead)",
            "q1b": "Price Perceptions Quantiles (Last 12 Months - q1b)",
            "q1c": "Price Perceptions Quantiles (Last 12 Months - q1c)"
        }

        fig = px.line(
            result_pd,
            x="date",
            y=["5th", "10th", "25th", "50th", "75th", "90th", "95th"],
            title=question_titles.get(question, f"{question} Quantiles"),
            markers=True,
        )

        # Update with reasonable labels
        fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=10),
                tickangle=30,  # Try 0 (default), 30, or 45 degrees
                title="Price perception/expectation",
            )
        )

        fig.show()

        # Convert the ordinal values to z-scores for analysis
        quantiles = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"]
        quantile_data = result_pd[quantiles]
        quantile_zscores = (
            quantile_data - quantile_data.mean()
        ) / quantile_data.std()

        # Add date back in for plotting
        quantile_zscores["date"] = result_pd["date"]

        # Melt the z-score dataframe for plotting
        melted_z = quantile_zscores.melt(
            id_vars="date", var_name="Quantile", value_name="Z-score"
        )

        # Create the z-score line plot
        fig_zscore = go.Figure()

        for quantile in melted_z["Quantile"].unique():
            subset = melted_z[melted_z["Quantile"] == quantile]
            fig_zscore.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["Z-score"],
                    mode="lines",
                    name=quantile,
                )
            )

        fig_zscore.update_layout(
            title=f"Standardized {question_titles.get(question, question)} by Quantile",
            xaxis_title="Date",
            yaxis_title="Z-score",
            legend_title="Quantile",
            template="plotly_white",
        )

        fig_zscore.show()

        ### WRITE TO EXCEL
        output_excel = f"inflation_attitudes_quantiles_discrete_{question}.xlsx"
        result_pd.to_excel(output_excel, index=False)
    return (quantile_maker,)


@app.cell
def _(quantile_maker):
    # 1 Year ahead
    quantile_maker("q2a_agg1")
    return


@app.cell
def _(quantile_maker):
    # 2 year ahead

    quantile_maker("q2b_agg1")
    return


@app.cell
def _(quantile_maker):
    # 5 Year ahead
    quantile_maker("q2c_agg1")
    return


@app.cell
def _(quantile_maker):
    # Price perceptions q1b
    quantile_maker("q1b")
    return


@app.cell
def _(quantile_maker):
    # Price perceptions q1c
    quantile_maker("q1c")
    return


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
        # "../ias_analysis/data/individual-responses-xlsx.xlsx",
        # "../ias_analysis/data/Inflation Attitudes Survey Feb 2025.xlsx",
        "data/individual-responses.xlsx",
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
         1: "Down by 5% or more",
         2: "Down by 4% but less than 5%",
         3: "Down by 3% but less than 4%",
         4: "Down by 2% but less than 3%",
         5: "Down by 1% but less than 2%",
         6: "Down by 1% or less",
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
    output_excel = "inflation_attitudes_quantiles_discrete_1yearahead.xlsx"
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
