

import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium", auto_download=["html"])


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
    # Load all required columns at once
    ias_raw = pl.read_excel(
        "data/Inflation Attitudes Survey Feb 2025.xlsx",
        sheet_name="Dataset",
        columns=[
            "weight", "yyyyqq", "age", 
            "work", "class", "tenure", "income", "sreg",
            "q2a_agg1", "q2b_agg1", "q2c_agg1", "q2aiii",
            "q17_1", "q17_2", "q17_3", "q17_4", "q17_5", "q17_6", "q17_7", "q17_8"
        ]
    )
    return (ias_raw,)


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
def _(Dict, go, pd):
    def plot_question_medians(df: pd.DataFrame, label_mapping: Dict[str, str] | None = None, title: str = "Grouped Median"):
        # Convert to Pandas for Plotly compatibility
        pdf = df.to_pandas()

        # Rename columns
        if label_mapping:
            pdf = pdf.rename(columns={code: label for code, label in label_mapping.items() if code in pdf.columns})

        # Melt the DataFrame to long format for easier plotting
        df_melted = pdf.melt(id_vars="yyyyqq", var_name="Group", value_name="Median")

        # Create Plotly line chart
        fig = go.Figure()

        for grp in sorted(df_melted["Group"].unique()):
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
        return fig
    return (plot_question_medians,)


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

    ### THIS IS WHERE THE CALCULATION LOGIC SITS
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
def _():
    # Mapping dictionaries for different categories
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

    age_map = {
        "1": "15-24",
        "2": "25-34",
        "3": "35-44",
        "4": "45-54",
        "5": "55-64",
        "6": "65+"
    }

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

    housing_map = {
        "1": "Owned outright",
        "2": "Mortgage",
        "3": "Council Rent",
        "4": "Other"
    }

    income_map = {
        "1": "<9500 [option removed 2022 Feb]",
        "2": "9500-17499 [option removed 2022 Feb]",
        "3": "17500-24999 [option removed 2022 Feb]",
        "4": ">25000 [option removed from 2016 Feb]",
        "5": "25000-39999 [option added 2016 Feb, option removed 2022 Feb]",
        "6": ">40000 [option added 2016 Feb, option removed 2022 Feb]",
        "7": "<9999 [option added 2022 Feb]",
        "8": "10000-19999 [option added 2022 Feb]",
        "9": "20000-34999 [option added 2022 Feb]",
        "10": "35000-44999 [option added 2022 Feb]",
        "11": ">45000 [option added 2022 Feb]",
        "12": "Prefer not to answer [option added 2022 Feb]"
    }
    return (
        age_map,
        class_mapping,
        emp_map,
        housing_map,
        income_map,
        q2_agg_ias_class_bounds,
        sreg_bands,
    )


@app.cell
def _(
    clean_ias,
    clean_ias_generic,
    comp_grouped_medians,
    ias_raw,
    q2_agg_ias_class_bounds,
):
    # Create all analyses from the same base DataFrame
    analyses = {
        'age': clean_ias_generic(ias_raw, 'age'),
        'work': clean_ias_generic(ias_raw, 'work'),
        'class': clean_ias_generic(ias_raw, 'class'),
        'tenure': clean_ias_generic(ias_raw, 'tenure'),
        'income': clean_ias_generic(ias_raw, 'income'),
        'sreg': clean_ias_generic(ias_raw, 'sreg')
    }

    ias_clean = clean_ias(ias_raw)

    # Compute medians for each analysis
    median_results = {
        category: comp_grouped_medians(df, q2_agg_ias_class_bounds, disagg_col=category)
        for category, df in analyses.items()
    }
    return analyses, ias_clean, median_results


@app.cell
def _(mo):
    mo.md(r"""## IE by Age""")
    return


@app.cell
def _(age_map, median_results, plot_question_medians):
    plot_question_medians(median_results['age'][0], age_map, "1 year ahead inflation expectations by age")
    plot_question_medians(median_results['age'][1], age_map, "2 year ahead inflation expectations by age")
    plot_question_medians(median_results['age'][2], age_map, "5 year ahead inflation expectations by age")
    return


@app.cell
def _(mo):
    mo.md(r"""## IE by employment status""")
    return


@app.cell
def _(emp_map, median_results, plot_question_medians):
    plot_question_medians(median_results['work'][0], emp_map, "1 year ahead inflation expectations by employment")
    plot_question_medians(median_results['work'][1], emp_map, "2 year ahead inflation expectations by employment")
    plot_question_medians(median_results['work'][2], emp_map, "5 year ahead inflation expectations by employment")
    return


@app.cell
def _():
    return


@app.cell
def _(class_mapping, median_results, plot_question_medians):
    plot_question_medians(median_results['class'][0], class_mapping, "1 year ahead inflation expectations by class")
    plot_question_medians(median_results['class'][1], class_mapping, "2 year ahead inflation expectations by class")
    plot_question_medians(median_results['class'][2], class_mapping, "5 year ahead inflation expectations by class")
    return


@app.cell
def _(mo):
    mo.md(r"""## IE by housing tenure""")
    return


@app.cell
def _(housing_map, median_results, plot_question_medians):
    plot_question_medians(median_results['tenure'][0], housing_map, "1 year ahead inflation expectations by tenure")
    plot_question_medians(median_results['tenure'][1], housing_map, "2 year ahead inflation expectations by tenure")
    plot_question_medians(median_results['tenure'][2], housing_map, "5 year ahead inflation expectations by tenure")
    return


@app.cell
def _(mo):
    mo.md(r"""## IE by income""")
    return


@app.cell
def _(income_map, median_results, plot_question_medians):
    plot_question_medians(median_results['income'][0], income_map, "1 year ahead inflation expectations by income")
    plot_question_medians(median_results['income'][1], income_map, "2 year ahead inflation expectations by income")
    plot_question_medians(median_results['income'][2], income_map, "5 year ahead inflation expectations by income")
    return


@app.cell
def _(mo):
    mo.md(r"""## IE by region (sreg)""")
    return


@app.cell
def _(median_results, plot_question_medians, sreg_bands):
    plot_question_medians(median_results['sreg'][0], sreg_bands, "1 year ahead inflation expectations by sreg")
    plot_question_medians(median_results['sreg'][1], sreg_bands, "2 year ahead inflation expectations by sreg")
    plot_question_medians(median_results['sreg'][2], sreg_bands, "5 year ahead inflation expectations by sreg")
    return


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Don't Knows by Different variables

        Steps taken to adjust for weights:

        - For each timestamp, for a particular group (e.g. an age group)
        - We calculate the total sum of weights
        - Then we calculate the total sum of the weights for don't knows, for that particular group
        - Then we divide the sum of weights for the don't knows in the category by the total don't knows
        """
    )
    return (mo,)


@app.cell
def _(convert_yyyyqq_to_datetime, ias_clean, pl):
    ### THIS IS WHERE THE CALCULATION LOGIC SITS
    def calc_dk_proportions_by_category(df, category_col="age"):
        """Calculate Don't Know proportions for all horizons for a given category"""

        def calc_single_horizon(question_col):
            # Calculate total weights by category and timestamp
            total_weights = (
                df
                .group_by(["yyyyqq", category_col])
                .agg(pl.col("weight").sum().alias("total_weight"))
            )

            # Calculate weights for all "Don't Know" responses for a question, grouped by timestamp and category (e.g. age)
            dk_weights = (
                df
                .filter(pl.col(question_col) == "19")
                .group_by(["yyyyqq", category_col])
                .agg(pl.col("weight").sum().alias("dk_weight"))
            )

            # Join and calculate proportions (for each timestamp, divide the weight of the category's don't knows by the weight of the category's total responses)
            proportions = (
                dk_weights
                .join(total_weights, on=["yyyyqq", category_col], how="right")
                .with_columns([
                    (pl.col("dk_weight") / pl.col("total_weight")).alias("proportion")
                ])
                .fill_null(0)
                .sort(["yyyyqq", category_col])
            )

            # Add datetime column and pivot
            proportions = proportions.with_columns([
                pl.col("yyyyqq").map_elements(lambda x: convert_yyyyqq_to_datetime(str(x))).alias("date")
            ])

            props_pivoted = proportions.pivot(
                values="proportion",
                index="date",
                on=category_col,
                aggregate_function="first"
            ).sort("date")

            return proportions, props_pivoted.to_pandas()

        # Calculate for all three horizons
        results = {
            '1yr': calc_single_horizon("q2a_agg1"),
            '2yr': calc_single_horizon("q2b_agg1"),
            '5yr': calc_single_horizon("q2c_agg1")
        }

        return results

    # Calculate DK proportions for each category
    dk_results = {
        'age': calc_dk_proportions_by_category(ias_clean, "age"),
        'work': calc_dk_proportions_by_category(ias_clean, "work"),
        'class': calc_dk_proportions_by_category(ias_clean, "class"),
        'income': calc_dk_proportions_by_category(ias_clean, "income"),
        'sreg': calc_dk_proportions_by_category(ias_clean, "sreg")
    }
    return calc_dk_proportions_by_category, dk_results


@app.cell
def _(
    Dict,
    age_map,
    class_mapping,
    dk_results,
    emp_map,
    go,
    income_map,
    pd,
    sreg_bands,
):
    def plot_dk_proportions(df: pd.DataFrame, label_mapping: Dict[str, str] | None = None, title: str = "Don't Know Responses"):
        fig = go.Figure()

        for col in sorted(df.columns):
            if col != "date":  # Skip the datetime column
                name = label_mapping.get(str(col), str(col))
                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=df[col],
                    mode='lines+markers',
                    name=name
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Quarter",
            yaxis_title="Proportion",
            legend_title="Group",
            template="plotly_white",
            hovermode="x unified"
        )

        fig.show()
        return fig

    # Plot DK proportions for all categories and horizons
    horizons = {
        '1yr': "1 year ahead",
        '2yr': "2 years ahead",
        '5yr': "5 years ahead"
    }

    category_titles = {
        'age': 'by age',
        'work': 'by employment status',
        'class': 'by class status',
        'income': 'by income',
        'sreg': 'by region'
    }

    category_maps = {
        'age': age_map,
        'work': emp_map,
        'class': class_mapping,
        'income': income_map,
        'sreg': sreg_bands
    }

    dk_figures = {}
    for cat, res in dk_results.items():
        dk_figures[cat] = {}
        for hrz, (_, res_df) in res.items():
            title = f"Proportion of 'Don't Know' responses {horizons[hrz]} {category_titles[cat]}"
            fig = plot_dk_proportions(
                res_df,
                label_mapping=category_maps.get(cat),
                title=title
            )
            dk_figures[cat][hrz] = fig
    return (
        cat,
        category_maps,
        category_titles,
        dk_figures,
        fig,
        horizons,
        hrz,
        plot_dk_proportions,
        res,
        res_df,
        title,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Alternate weight adjustment (CORRECT VERSION)
        To correctly adjust for weights, we must:
        - For each don't know, their vote isn't = 1. It needs to be adjusted by their weighting within the timestamp.
        - This means dividing their value by the total weight
        This is a simple weighting schema, no need to overcomplicate.
        """
    )
    return


@app.cell
def _(
    category_maps,
    category_titles,
    convert_yyyyqq_to_datetime,
    horizons,
    ias_clean,
    pl,
    plot_dk_proportions,
):
    def calc_weight_adjusted_dk_by_cat(df, category_col="age"):

        def calc_single_horizon(question_col):
            # Calculate weights for all "Don't Know" responses for a question, grouped by timestamp and category (e.g. age)
            dk_weights = (
                df
                .filter(pl.col(question_col) == "19")
                .group_by(["yyyyqq", category_col])
                .agg(pl.col("weight").sum().alias("dk_weight"))
            )

            # Calculate the number of respondents overall for that timestamp
            total_weights = (
                df
                .group_by("yyyyqq")
                .agg(pl.col("weight").sum().alias("total_weights"))        
            )

            # Divide the total weight in the category & stamp by the number of respondents
            adj_respondent_num = (
                dk_weights
                .join(total_weights, on=["yyyyqq"], how="right")
                .with_columns([
                    (pl.col("dk_weight") / pl.col("total_weights")).alias("adj_respondent_num")
                ])
                .fill_null(0)
                .sort(["yyyyqq", category_col])
            )

            # Add datetime column and pivot
            adj_respondent_num = adj_respondent_num.with_columns([
                pl.col("yyyyqq").map_elements(lambda x: convert_yyyyqq_to_datetime(str(x))).alias("date")
            ])

            adj_respondent_num_pivoted = adj_respondent_num.pivot(
                values="adj_respondent_num",
                index="date",
                on=category_col,
                aggregate_function="first"
            ).sort("date")

            # Define the indexation date for 2020q1 using the same conversion function
            baseline_date = convert_yyyyqq_to_datetime("202001")

            # Convert to pandas DataFrame for further manipulation
            res_df = adj_respondent_num_pivoted.to_pandas()

            # For each column (other than 'date'), index the series to its value at baseline_date.
            for col in res_df.columns:
                if col == 'date':
                    continue

                ##### ------------->>>>> EDIT HERE TO CHANGE THE INDEXATION <<<<<------------
                # # Find the baseline value at 2020q1
                # baseline_val = res_df.loc[res_df['date'] == baseline_date, col].iloc[0]

                # # Index each value by dividing by the baseline value
                # res_df[col] = res_df[col] / baseline_val

            return adj_respondent_num_pivoted, res_df

        # Calculate for all three horizons
        results = {
            '1yr': calc_single_horizon("q2a_agg1"),
            '2yr': calc_single_horizon("q2b_agg1"),
            '5yr': calc_single_horizon("q2c_agg1")
        }

        return results

    # Calculate weight adjusted DK numbers for each category
    weight_adjusted_dk_numbers = {
        'age': calc_weight_adjusted_dk_by_cat(ias_clean, "age"),
        'work': calc_weight_adjusted_dk_by_cat(ias_clean, "work"),
        'class': calc_weight_adjusted_dk_by_cat(ias_clean, "class"),
        'income': calc_weight_adjusted_dk_by_cat(ias_clean, "income"),
        'sreg': calc_weight_adjusted_dk_by_cat(ias_clean, "sreg")
    }

    def _():
        weight_adj_dk_figs = {}
        for cat, res in weight_adjusted_dk_numbers.items():
            weight_adj_dk_figs[cat] = {}
            for hrz, (_, res_df) in res.items():
                title = f"Weight adjusted proportion of 'Don't Know' responses: {horizons[hrz]} {category_titles[cat]}"
                fig = plot_dk_proportions(
                    res_df,
                    label_mapping=category_maps.get(cat),
                    title=title
                )
                weight_adj_dk_figs[cat][hrz] = fig
        return


    _()
    return calc_weight_adjusted_dk_by_cat, weight_adjusted_dk_numbers


@app.cell
def _(mo):
    mo.md(
        r"""
        # Consumption and spending reasoning by rising/falling IE
        - For 1 year ahead expectations
        - Take those who's expectations have either gone up ('risen a lot', 'risen a little') or gone down ('fallen a lot', 'fallen a little') in response to question 2aiii
        - For each of these two categories, create a time series for the proportion of responses to question 17 (the planning question).
        """
    )
    return


@app.cell
def _(convert_yyyyqq_to_datetime, go, ias_clean, pd, pl):
    q2a_map_2 = {
        '1': 'Risen a lot',
        '2': 'Risen a little',
        '3': 'Unchanged',
        '4': 'Fallen a little',
        '5': 'Fallen a lot',
        '6': 'No idea'
    }

    def q17_responses_by_up_or_down2(df):
        # Adjust our date column
        df = df.with_columns([
            pl.col("yyyyqq")
              .map_elements(lambda x: convert_yyyyqq_to_datetime(str(x)))
              .alias("date")
        ])

        # Columns we want to cast from string to int
        q17_cols = [f"q17_{i}" for i in range(1, 9)]

        # Cast q17 columns to integers (from string "0"/"1")
        df = df.with_columns([
            pl.col(col).cast(pl.Int64).alias(col) for col in q17_cols
        ])

        # Split expectations
        expectations_up = df.filter(pl.col('q2aiii').is_in(["1", "2"]))
        expectations_down = df.filter(pl.col('q2aiii').is_in(["4", "5"]))

        # Sum q17 responses for upward expectations
        q17_up = expectations_up.group_by("date").agg([
            pl.sum(col).alias(f"{col}_up") for col in q17_cols
        ]).sort("date")

        # Sum q17 responses for downward expectations
        q17_down = expectations_down.group_by("date").agg([
            pl.sum(col).alias(f"{col}_down") for col in q17_cols
        ]).sort("date")

        # Plot upward expectations
        fig_up = go.Figure()
        for col in q17_up.columns:
            if col != "date":
                fig_up.add_trace(go.Scatter(
                    x=q17_up["date"].to_list(),
                    y=q17_up[col].to_list(),
                    mode="lines",
                    name=col
                ))
        fig_up.update_layout(
            title="q17 Responses Over Time - Expectations Up",
            xaxis_title="Date",
            yaxis_title="Sum of Responses"
        )
        fig_up.show()

        # Plot downward expectations
        fig_down = go.Figure()
        for col in q17_down.columns:
            if col != "date":
                fig_down.add_trace(go.Scatter(
                    x=q17_down["date"].to_list(),
                    y=q17_down[col].to_list(),
                    mode="lines",
                    name=col
                ))
        fig_down.update_layout(
            title="q17 Responses Over Time - Expectations Down",
            xaxis_title="Date",
            yaxis_title="Sum of Responses"
        )
        fig_down.show()

        print(expectations_up.tail())

        # Write to excel
        # Convert Polars DataFrames to Pandas DataFrames
        q17_up_pd = q17_up.to_pandas()
        q17_down_pd = q17_down.to_pandas()

        # Write the DataFrames to Excel, each on its own sheet
        with pd.ExcelWriter("q17_responses.xlsx") as writer:
            q17_up_pd.to_excel(writer, sheet_name="Q17_Up", index=False)
            q17_down_pd.to_excel(writer, sheet_name="Q17_Down", index=False)


    # Call the function with your dataset
    q17_responses_by_up_or_down2(ias_clean)
    return q17_responses_by_up_or_down2, q2a_map_2


@app.cell
def _(mo):
    mo.md(r"""# Write to Excel""")
    return


@app.cell
def _(category_maps, dk_results, pd):
    # Create Excel writer object
    def dk_write_wrapper():
        with pd.ExcelWriter("dont_knows_by_category.xlsx") as writer:
            for category, results in dk_results.items():
                for horizon, (_, df) in results.items():
                    # Get the appropriate mapping if it exists
                    mapping = category_maps.get(category, {})
                    if mapping:
                        df = df.rename(columns={col: mapping.get(str(col), str(col)) for col in df.columns if col != 'date'})

                    # Write to Excel with sheet name combining category and horizon
                    sheet_name = f"{category}_{horizon}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

    dk_write_wrapper()
    return (dk_write_wrapper,)


@app.cell
def _(category_maps, median_results, pd):
    # Create Excel writer object
    def meds_write_wrapper():
        with pd.ExcelWriter("inflation_expectations_by_category.xlsx") as writer:
            horizons = {
                0: "1yr_ahead",
                1: "2yr_ahead",
                2: "5yr_ahead"
            }

            for category, results in median_results.items():
                # results is a tuple of three DataFrames (1yr, 2yr, 5yr)
                for i, df in enumerate(results):
                    # Convert to pandas
                    pdf = df.to_pandas()

                    # Get the appropriate mapping if it exists
                    mapping = category_maps.get(category, {})
                    if mapping:
                        pdf = pdf.rename(columns={
                            col: mapping.get(str(col), str(col)) 
                            for col in pdf.columns 
                            if col != 'yyyyqq'
                        })

                    # Write to Excel with sheet name combining category and horizon
                    sheet_name = f"{category}_{horizons[i]}"
                    pdf.to_excel(
                        writer, 
                        sheet_name=sheet_name, 
                        index=False
                    )

    meds_write_wrapper()
    return (meds_write_wrapper,)


@app.cell
def _(mo):
    mo.md(r"""## Write the DK by category weight-adjusted counts""")
    return


@app.cell
def _(pd, weight_adjusted_dk_numbers):
    def write_results_to_excel2(results, filename="analysis_results.xlsx"):
        """
        Writes analysis results to an Excel workbook with each category on its own sheet.
        For each category, the three horizons ('1yr', '2yr', '5yr') are merged horizontally
        based on the 'date' column. The columns (except the date) are renamed using the provided
        mapping dictionaries, and a multi-index is created on the columns so that the top level
        is the horizon and the second level is the informative column name.

        Parameters:
          results (dict): Dictionary of analysis results (e.g. weight_adjusted_dk_numbers).
                          Expected structure:
                            { category: { horizon: (pivot_df, indexed_df), ... }, ... }
          filename (str): Name of the Excel file to write.
        """
        # Mapping dictionaries by category:
        mapping_dict = {
            'age': {
                "1": "15-24",
                "2": "25-34",
                "3": "35-44",
                "4": "45-54",
                "5": "55-64",
                "6": "65+"
            },
            'work': {
                "1": "Full or Part Time",
                "2": "Unemployed"
            },
            'class': {
                "1": "AB",
                "2": "C1",
                "3": "C2",
                "4": "DE"    
            },
            'income': {
                "1": "<9500 [option removed 2022 Feb]",
                "2": "9500-17499 [option removed 2022 Feb]",
                "3": "17500-24999 [option removed 2022 Feb]",
                "4": ">25000 [option removed from 2016 Feb]",
                "5": "25000-39999 [option added 2016 Feb, option removed 2022 Feb]",
                "6": ">40000 [option added 2016 Feb, option removed 2022 Feb]",
                "7": "<9999 [option added 2022 Feb]",
                "8": "10000-19999 [option added 2022 Feb]",
                "9": "20000-34999 [option added 2022 Feb]",
                "10": "35000-44999 [option added 2022 Feb]",
                "11": ">45000 [option added 2022 Feb]",
                "12": "Prefer not to answer [option added 2022 Feb]"
            },
            'sreg': {
                "1": "Scotland",
                "2": "North & NI",
                "3": "Midlands",
                "4": "Wales and West",
                "5": "South East"
            }
        }

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Loop over each category (each will be a separate sheet)
            for cat, horizon_dict in results.items():
                mapping = mapping_dict.get(cat, {})  # Get the mapping for the category, if available
                merged_df = None

                # Process each horizon's results
                for hrz, (pivot_df, indexed_df) in horizon_dict.items():
                    df_temp = indexed_df.copy()
                    # Set 'date' as the index so that merging is on date
                    df_temp = df_temp.set_index("date")
                    # Rename columns using the provided mapping. If a key isn't found, keep original.
                    df_temp.rename(columns=lambda col: mapping.get(col, col), inplace=True)
                    # Create a MultiIndex for columns: top level is the horizon label, second level is the (mapped) column name.
                    df_temp.columns = pd.MultiIndex.from_product([[hrz], df_temp.columns])

                    if merged_df is None:
                        merged_df = df_temp
                    else:
                        merged_df = merged_df.join(df_temp, how='outer')

                # Sort the merged DataFrame by date index
                merged_df.sort_index(inplace=True)
                # Reset the index so that 'date' becomes a column again.
                merged_df = merged_df.reset_index()
                # For the 'date' column, create a two-level header where the top level is blank.
                cols = merged_df.columns.tolist()
                # The first column is 'date'
                cols[0] = ('', 'date')
                # The remaining columns are already MultiIndex; if any are not, convert them.
                new_cols = []
                for col in cols:
                    if isinstance(col, tuple):
                        new_cols.append(col)
                    else:
                        new_cols.append(('', col))
                merged_df.columns = pd.MultiIndex.from_tuples(new_cols)

                merged_df.index.name = "date"  # So it's labelled in Excel
                merged_df.to_excel(writer, sheet_name=cat, index=True)
                # Write the multi-indexed DataFrame to its own sheet
                # merged_df.to_excel(writer, sheet_name=cat, index=False)

        print(f"Results written to {filename}")

    write_results_to_excel2(weight_adjusted_dk_numbers, filename="weight_adjusted_dk_by_category.xlsx")
    return (write_results_to_excel2,)


if __name__ == "__main__":
    app.run()
