import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Inflation Expectations by categories""")
    return


@app.cell
def _():
    import polars as pl
    import plotly.graph_objects as go
    import pandas as pd
    from typing import Dict, Tuple, Optional
    return Dict, Optional, Tuple, go, pd, pl


@app.cell
def _(pl):
    ias_raw = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=["weight", "yyyyqq", 'age', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    ias_raw.head()
    return (ias_raw,)


@app.cell
def _():
    q2_agg_ias_class_bounds = {
        "1": (-1.0, 1.0),   # Down by 1% or less
        "2": (-2.0, 1.0),   # Down by 1% to <2%
        "3": (-3.0, 1.0),
        "4": (-4.0, 1.0),
        "5": (-5.0, 1.0),
        "6": (-6.0, 1.0),   # Down by 5% or more (approx.)
        "7": (0.0, 0.0),    # Not changed
        "8": (0.0, 1.0),
        "9": (1.0, 1.0),
        "10": (2.0, 1.0),
        "11": (3.0, 1.0),
        "12": (4.0, 1.0),
        "13": (5.0, 1.0),
        "14": (6.0, 1.0),
        "15": (7.0, 1.0),
        "16": (8.0, 1.0),
        "17": (9.0, 1.0),
        "18": (10.0, 1.0),  # Up by 10% or more (approx.)
        "20": (10.0, 1.0),  # Up by 10% to <11%
        "21": (11.0, 1.0),
        "22": (12.0, 1.0),
        "23": (13.0, 1.0),
        "24": (14.0, 1.0),
        "25": (15.0, 5.0),  # Up by 15% or more (extended width)
    }
    return (q2_agg_ias_class_bounds,)


@app.cell
def _(Dict, Optional, Tuple, pd, pl):
    def clean_ias(df: pl.DataFrame): 
        df = df.filter(pl.col("q2a_agg1").is_not_null())
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

    def grouped_median_unequal_widths(counts: Dict[str, int], class_bounds: Dict[str, Tuple[float, float]]) -> Optional[float]:
        """
        Interpolated grouped median with variable class widths.

        counts: dict of {category_label: count}
        class_bounds: dict of {category_label: (lower_bound, width)}
        Returns interpolated median as float, or None if no data.
        """
        if not counts:
            return None

        # Sort by lower bound of the class
        sorted_items = sorted(
            [(label, class_bounds[label][0], class_bounds[label][1], count)
             for label, count in counts.items() if label in class_bounds],
            key=lambda x: x[1]
        )

        total = sum(item[3] for item in sorted_items)
        if total == 0:
            return None

        median_pos = total / 2
        cum = 0
        for label, L, w, f in sorted_items:
            prev_cum = cum
            cum += f
            if cum >= median_pos:
                return L + ((median_pos - prev_cum) / f) * w
        return None

    def compute_grouped_medians_polars(df: pl.DataFrame, class_bounds: Dict[str, Tuple[float, float]]):
        questions = ['q2a_agg1', 'q2b_agg1', 'q2c_agg1']
        result_frames = {q: {} for q in questions}

        # Group by timestamp and age
        grouped = df.group_by(["yyyyqq", "age"])

        for (yyyyqq, age), group in grouped:
            for question in questions:
                counts_df = group[question].value_counts()
                if counts_df.height == 0:
                    median = None
                else:
                    # Convert to dictionary
                    counts = dict(zip(counts_df[question], counts_df["count"]))
                    median = grouped_median_unequal_widths(counts, class_bounds)

                result_frames[question].setdefault(yyyyqq, {})[age] = median

        # Convert results into DataFrames
        output = {}
        for question, data in result_frames.items():
            rows = []
            for yyyyqq, age_dict in data.items():
                row = {
                    "yyyyqq": convert_yyyyqq_to_datetime(str(yyyyqq))
                }
                row.update({str(age): val for age, val in age_dict.items()})
                rows.append(row)
            output[question] = pl.DataFrame(rows).sort("yyyyqq")

        return output['q2a_agg1'], output['q2b_agg1'], output['q2c_agg1']
    return (
        clean_ias,
        compute_grouped_medians_polars,
        convert_yyyyqq_to_datetime,
        grouped_median_unequal_widths,
    )


@app.cell
def _(
    clean_ias,
    compute_grouped_medians_polars,
    ias_raw,
    q2_agg_ias_class_bounds,
):
    ias_clean = clean_ias(ias_raw)
    q2a_median_by_age, q2b_median_by_age, q2c_median_by_age = compute_grouped_medians_polars(ias_clean, q2_agg_ias_class_bounds)
    return ias_clean, q2a_median_by_age, q2b_median_by_age, q2c_median_by_age


@app.cell
def _(Dict, go, pd):
    def plot_question_medians(df: pd.DataFrame, label_mapping: Dict[str, str] | None = None, title: str = "Grouped Median by Age"):
        # Convert to Pandas for Plotly compatibility
        pdf = df.to_pandas()

        # Rename age columns
        if label_mapping:
            pdf = pdf.rename(columns={code: label for code, label in label_mapping.items() if code in pdf.columns})

        # Melt the DataFrame to long format for easier plotting
        df_melted = pdf.melt(id_vars="yyyyqq", var_name="Group", value_name="Median")

        # Create Plotly line chart
        fig = go.Figure()

        for grp in df_melted["Group"].unique():
            group_data = df_melted[df_melted["Group"] == grp]
            fig.add_trace(go.Scatter(
                x=group_data["yyyyqq"],
                y=group_data["Median"],
                mode='lines+markers',
                name=grp
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Quarter",
            yaxis_title="Grouped Median (Interpolated)",
            legend_title="Group",
            template="plotly_white",
            hovermode="x unified"
        )

        fig.show()

    age_map = {
        "1": "15-24",
        "2": "25-34",
        "3": "35-44",
        "4": "45-54",
        "5": "55-64",
        "6": "65+"
    }
    return age_map, plot_question_medians


@app.cell
def _(age_map, plot_question_medians, q2a_median_by_age):
    plot_question_medians(q2a_median_by_age, age_map, title="Year ahead inflation expectations by age")
    return


@app.cell
def _(age_map, plot_question_medians, q2b_median_by_age):
    plot_question_medians(q2b_median_by_age, age_map, title="2-year ahead inflation expectations by age")
    return


@app.cell
def _(age_map, plot_question_medians, q2c_median_by_age):
    plot_question_medians(q2c_median_by_age, age_map, title="5-Year ahead inflation expectations by age")
    return


@app.cell
def _(
    Dict,
    Tuple,
    convert_yyyyqq_to_datetime,
    grouped_median_unequal_widths,
    pl,
):
    def clean_ias_generic(df: pl.DataFrame, col: str = "age", mapping: Dict[str, str] | None = None): 
        """
        Filters the DataFrame to rows where 'q2a_agg1' is not null, and optionally recodes
        the specified column (default "age") using the provided mapping.

        Parameters:
          df (pl.DataFrame): Input DataFrame.
          col (str): The name of the column to recode.
          mapping (dict): A dictionary where keys are original values and values are new codes.

        Returns:
          pl.DataFrame: Cleaned DataFrame.
        """
        df = df.filter(pl.col("q2a_agg1").is_not_null())

        if mapping is not None:
            # Build a conditional expression for recoding
            # Start with the first mapping condition:
            keys = list(mapping.keys())
            expr = pl.when(pl.col(col) == keys[0]).then(mapping[keys[0]])
            for key in keys[1:]:
                expr = expr.when(pl.col(col) == key).then(mapping[key])
            expr = expr.otherwise(pl.col(col))
            df = df.with_columns([expr.alias(col)])

        return df

    def comp_grouped_medians(
        df: pl.DataFrame,
        class_bounds: Dict[str, Tuple[float, float]],
        disagg_col: str = "age"
    ):
        """
        Compute grouped medians for a set of question columns by timestamp and any disaggregation column.

        Parameters:
          df (pl.DataFrame): Input DataFrame.
          class_bounds (dict): Mapping from category labels to (lower_bound, width) tuples.
          disagg_col (str): Column name to disaggregate by (default is "age").

        Returns:
          Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: DataFrames for 'q2a_agg1', 'q2b_agg1', and 'q2c_agg1'.
        """
        questions = ['q2a_agg1', 'q2b_agg1', 'q2c_agg1']
        result_frames = {q: {} for q in questions}

        # Group by timestamp and the specified disaggregation column
        grouped = df.group_by(["yyyyqq", disagg_col])

        for (yyyyqq, category), group in grouped:
            for question in questions:
                counts_df = group[question].value_counts()
                if counts_df.height == 0:
                    median = None
                else:
                    # Convert to dictionary: category value -> count
                    counts = dict(zip(counts_df[question], counts_df["count"]))
                    median = grouped_median_unequal_widths(counts, class_bounds)

                result_frames[question].setdefault(yyyyqq, {})[category] = median

        # Convert results into DataFrames, one per question
        output = {}
        for question, data in result_frames.items():
            rows = []
            for yyyyqq, cat_dict in data.items():
                row = {"yyyyqq": convert_yyyyqq_to_datetime(str(yyyyqq))}
                # Each key in the row is now a string representation of the disaggregation category
                row.update({str(cat): val for cat, val in cat_dict.items()})
                rows.append(row)
            output[question] = pl.DataFrame(rows).sort("yyyyqq")

        return output['q2a_agg1'], output['q2b_agg1'], output['q2c_agg1']
    return clean_ias_generic, comp_grouped_medians


@app.cell
def _(clean_ias_generic, comp_grouped_medians, pl, q2_agg_ias_class_bounds):
    # Employment status analysis
    ias_raw_work = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=['weight', 'yyyyqq', 'work', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    df_clean_work = clean_ias_generic(ias_raw_work, 'work')
    median_dfs_work = comp_grouped_medians(df_clean_work, q2_agg_ias_class_bounds, disagg_col="work")

    # Class status analysis
    ias_raw_class = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=['weight', 'yyyyqq', 'class', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    df_clean_class = clean_ias_generic(ias_raw_class, 'class')
    median_dfs_class = comp_grouped_medians(df_clean_class, q2_agg_ias_class_bounds, disagg_col="class")

    # Housing tenure analysis
    ias_raw_tenure = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=['yyyyqq', 'tenure', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    df_clean_tenure = clean_ias_generic(ias_raw_tenure, 'tenure')
    median_dfs_tenure = comp_grouped_medians(df_clean_tenure, q2_agg_ias_class_bounds, disagg_col="tenure")

    # Income analysis
    ias_raw_income = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=['yyyyqq', 'income', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    df_clean_income = clean_ias_generic(ias_raw_income, 'income')
    median_dfs_income = comp_grouped_medians(df_clean_income, q2_agg_ias_class_bounds, disagg_col="income")

    # Region analysis
    ias_raw_sreg = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=['yyyyqq', 'sreg', 'q2a_agg1', 'q2b_agg1', 'q2c_agg1']
    )
    df_clean_sreg = clean_ias_generic(ias_raw_sreg, 'sreg')
    median_dfs_sreg = comp_grouped_medians(df_clean_sreg, q2_agg_ias_class_bounds, disagg_col="sreg")
    return (
        df_clean_class,
        df_clean_income,
        df_clean_sreg,
        df_clean_tenure,
        df_clean_work,
        ias_raw_class,
        ias_raw_income,
        ias_raw_sreg,
        ias_raw_tenure,
        ias_raw_work,
        median_dfs_class,
        median_dfs_income,
        median_dfs_sreg,
        median_dfs_tenure,
        median_dfs_work,
    )


@app.cell
def _():
    # Mapping dictionaries for different categories
    emp_map = {
        "1": "Full or Part Time",
        "2": "Unemployed"
    }

    class_mapping = {
        "1": "AB",
        "2": "C1",
        "3": "C2",
        "4": "DE"    
    }

    sreg_bands = {
        "1": "Scotland",
        "2": "North & NI",
        "3": "Midlands",
        "4": "Wales and West",
        "5": "South East"
    }
    return class_mapping, emp_map, sreg_bands


@app.cell
def _(emp_map, median_dfs_work, plot_question_medians):
    plot_question_medians(median_dfs_work[0], emp_map, '1 year ahead inflation expectations by employment status')
    return


@app.cell
def _(emp_map, median_dfs_work, plot_question_medians):
    plot_question_medians(median_dfs_work[1], emp_map, '2 year ahead inflation expectations by employment status')
    return


@app.cell
def _(emp_map, median_dfs_work, plot_question_medians):
    plot_question_medians(median_dfs_work[2], emp_map, '5 year ahead inflation expectations by employment status')
    return


@app.cell
def _(class_mapping, median_dfs_class, plot_question_medians):
    plot_question_medians(median_dfs_class[0], class_mapping, '1 year ahead inflation expectations by class status')
    return


@app.cell
def _(class_mapping, median_dfs_class, plot_question_medians):
    plot_question_medians(median_dfs_class[1], class_mapping, '2 year ahead inflation expectations by class status')
    return


@app.cell
def _(class_mapping, median_dfs_class, plot_question_medians):
    plot_question_medians(median_dfs_class[2], class_mapping, '5 year ahead inflation expectations by class status')
    return


@app.cell
def _(median_dfs_sreg, plot_question_medians, sreg_bands):
    plot_question_medians(median_dfs_sreg[0], sreg_bands, '1 year ahead inflation expectations by region')
    return


@app.cell
def _(median_dfs_sreg, plot_question_medians, sreg_bands):
    plot_question_medians(median_dfs_sreg[1], sreg_bands, '2 year ahead inflation expectations by region')
    return


@app.cell
def _(median_dfs_sreg, plot_question_medians, sreg_bands):
    plot_question_medians(median_dfs_sreg[2], sreg_bands, '5 year ahead inflation expectations by region')
    return


@app.cell
def _():
    import marimo as mo
    mo.md(r"""# Don't Knows by Age""")
    return (mo,)


@app.cell
def _(convert_yyyyqq_to_datetime, ias_clean, pl):
    # First, calculate total weights by age group and timestamp
    total_weights_1yr = (
        ias_clean
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("total_weight"))
    )

    # Calculate weights for "Don't Know" responses
    dk_by_age = (
        ias_clean
        .filter(pl.col("q2a_agg1") == "19")
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("dk_weight"))
    )

    # Join the two tables and calculate proportions
    dk_proportions = (
        dk_by_age
        .join(total_weights_1yr, on=["yyyyqq", "age"], how="right")
        .with_columns([
            (pl.col("dk_weight") / pl.col("total_weight")).alias("proportion")
        ])
        .fill_null(0)  # Fill NaN with 0 for age groups with no "Don't Know" responses
        .sort(["yyyyqq", "age"])
    )

    # Convert yyyyqq to datetime
    dk_proportions = dk_proportions.with_columns([
        pl.col("yyyyqq").map_elements(lambda x: convert_yyyyqq_to_datetime(str(x))).alias("date")
    ])

    # Pivot the data to create columns for each age group
    dk_props_pivoted = dk_proportions.pivot(
        values="proportion",
        index="date", 
        on="age",
        aggregate_function="first"
    ).sort("date")

    # Convert to pandas for plotting
    dk_props_pd_1yr = dk_props_pivoted.to_pandas()
    return (
        dk_by_age,
        dk_proportions,
        dk_props_pd_1yr,
        dk_props_pivoted,
        total_weights_1yr,
    )


@app.cell
def _(convert_yyyyqq_to_datetime, ias_clean, pl):
    # Same calculation for 2-year ahead expectations
    total_weights_2yr = (
        ias_clean
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("total_weight"))
    )

    dk_by_age_2yr = (
        ias_clean
        .filter(pl.col("q2b_agg1") == "19")
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("dk_weight"))
    )

    dk_proportions_2yr = (
        dk_by_age_2yr
        .join(total_weights_2yr, on=["yyyyqq", "age"], how="right")
        .with_columns([
            (pl.col("dk_weight") / pl.col("total_weight")).alias("proportion")
        ])
        .fill_null(0)
        .sort(["yyyyqq", "age"])
    )

    dk_proportions_2yr = dk_proportions_2yr.with_columns([
        pl.col("yyyyqq").map_elements(lambda x: convert_yyyyqq_to_datetime(str(x))).alias("date")
    ])

    dk_props_pivoted_2yr = dk_proportions_2yr.pivot(
        values="proportion",
        index="date", 
        on="age",
        aggregate_function="first"
    ).sort("date")

    dk_props_pd_2yr = dk_props_pivoted_2yr.to_pandas()
    return (
        dk_by_age_2yr,
        dk_proportions_2yr,
        dk_props_pd_2yr,
        dk_props_pivoted_2yr,
        total_weights_2yr,
    )


@app.cell
def _(convert_yyyyqq_to_datetime, ias_clean, pl):
    # Same calculation for 5-year ahead expectations
    total_weights_5yr = (
        ias_clean
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("total_weight"))
    )

    dk_by_age_5yr = (
        ias_clean
        .filter(pl.col("q2c_agg1") == "19")
        .group_by(["yyyyqq", "age"])
        .agg(pl.col("weight").sum().alias("dk_weight"))
    )

    dk_proportions_5yr = (
        dk_by_age_5yr
        .join(total_weights_5yr, on=["yyyyqq", "age"], how="right")
        .with_columns([
            (pl.col("dk_weight") / pl.col("total_weight")).alias("proportion")
        ])
        .fill_null(0)
        .sort(["yyyyqq", "age"])
    )

    dk_proportions_5yr = dk_proportions_5yr.with_columns([
        pl.col("yyyyqq").map_elements(lambda x: convert_yyyyqq_to_datetime(str(x))).alias("date")
    ])

    dk_props_pivoted_5yr = dk_proportions_5yr.pivot(
        values="proportion",
        index="date", 
        on="age",
        aggregate_function="first"
    ).sort("date")

    dk_props_pd_5yr = dk_props_pivoted_5yr.to_pandas()
    return (
        dk_by_age_5yr,
        dk_proportions_5yr,
        dk_props_pd_5yr,
        dk_props_pivoted_5yr,
        total_weights_5yr,
    )


@app.cell
def _(age_map, dk_props_pd_1yr, go):
    # Create the plot for 1-year ahead expectations
    fig_1yr = go.Figure()

    # Plot each age group
    for age_col_1yr in dk_props_pd_1yr.columns:
        if age_col_1yr != "date":  # Skip the datetime column
            fig_1yr.add_trace(go.Scatter(
                x=dk_props_pd_1yr["date"],
                y=dk_props_pd_1yr[age_col_1yr],
                mode='lines+markers',
                name=age_map[str(age_col_1yr)]
            ))

    fig_1yr.update_layout(
        title="Proportion of 'Don't Know' Responses Within Each Age Group (1 year ahead)",
        xaxis_title="Quarter",
        yaxis_title="Proportion",
        legend_title="Age Group",
        template="plotly_white",
        hovermode="x unified"
    )

    fig_1yr.show()
    return age_col_1yr, fig_1yr


@app.cell
def _(age_map, dk_props_pd_2yr, go):
    # Create the plot for 2-year ahead expectations
    fig_2yr = go.Figure()

    # Plot each age group
    for age_col_2yr in dk_props_pd_2yr.columns:
        if age_col_2yr != "date":  # Skip the datetime column
            fig_2yr.add_trace(go.Scatter(
                x=dk_props_pd_2yr["date"],
                y=dk_props_pd_2yr[age_col_2yr],
                mode='lines+markers',
                name=age_map[str(age_col_2yr)]
            ))

    fig_2yr.update_layout(
        title="Proportion of 'Don't Know' Responses Within Each Age Group (2 years ahead)",
        xaxis_title="Quarter",
        yaxis_title="Proportion",
        legend_title="Age Group",
        template="plotly_white",
        hovermode="x unified"
    )

    fig_2yr.show()
    return age_col_2yr, fig_2yr


@app.cell
def _(age_map, dk_props_pd_5yr, go):
    # Create the plot for 5-year ahead expectations
    fig_5yr = go.Figure()

    # Plot each age group
    for age_col_5yr in dk_props_pd_5yr.columns:
        if age_col_5yr != "date":  # Skip the datetime column
            fig_5yr.add_trace(go.Scatter(
                x=dk_props_pd_5yr["date"],
                y=dk_props_pd_5yr[age_col_5yr],
                mode='lines+markers',
                name=age_map[str(age_col_5yr)]
            ))

    fig_5yr.update_layout(
        title="Proportion of 'Don't Know' Responses Within Each Age Group (5 years ahead)",
        xaxis_title="Quarter",
        yaxis_title="Proportion",
        legend_title="Age Group",
        template="plotly_white",
        hovermode="x unified"
    )

    fig_5yr.show()
    return age_col_5yr, fig_5yr


@app.cell
def _(age_map, dk_props_pd_1yr, dk_props_pd_2yr, dk_props_pd_5yr, pd):
    # Create Excel writer object
    with pd.ExcelWriter("dont_knows_by_age.xlsx") as writer:
        # Rename columns using age_map for each DataFrame
        dk_1yr = dk_props_pd_1yr.rename(columns={col: age_map[str(col)] for col in dk_props_pd_1yr.columns if col != 'date'})
        dk_2yr = dk_props_pd_2yr.rename(columns={col: age_map[str(col)] for col in dk_props_pd_2yr.columns if col != 'date'})
        dk_5yr = dk_props_pd_5yr.rename(columns={col: age_map[str(col)] for col in dk_props_pd_5yr.columns if col != 'date'})

        # Write each DataFrame to a different sheet
        dk_1yr.to_excel(
            writer,
            sheet_name="1yr_ahead",
            index=True
        )

        dk_2yr.to_excel(
            writer, 
            sheet_name="2yr_ahead",
            index=True
        )

        dk_5yr.to_excel(
            writer,
            sheet_name="5yr_ahead", 
            index=True
        )
    return dk_1yr, dk_2yr, dk_5yr, writer


if __name__ == "__main__":
    app.run()
