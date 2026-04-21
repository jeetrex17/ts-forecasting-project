from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


COMMON_SETUP = """
            from pathlib import Path
            import sys

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / 'data').exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
"""


def notebook_00():
    return [
        md(
            """
            # 00. Project Map for `codex_notebooks`

            This notebook defines the rebuilt project story before we dive into EDA or models.

            **By the end of this notebook you should understand**
            - what the forecasting task actually is
            - why weighted skill, chronology, and split design matter
            - which series are used as the 4 main representatives
            - what the small broader benchmark is doing
            """
        ),
        code(
            f"""
            {COMMON_SETUP}

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            from codex_notebooks.support import (
                CODEX_DIR,
                apply_plot_style,
                build_codex_artifacts,
                export_figure,
                load_artifact,
                load_json_artifact,
            )

            apply_plot_style()
            build_codex_artifacts(force=False, verbose=True)

            overview = load_json_artifact('overview.json')
            chosen = load_artifact('chosen_manifest.parquet')
            benchmark = load_artifact('benchmark_manifest.parquet')
            """
        ),
        md(
            """
            ## 1. Project scope

            The rebuilt notebook layer is intentionally narrower than the repo README. It focuses on:
            - the panel forecasting problem itself
            - EDA that changes model choice
            - a clean classical study
            - a clean deep-sequence study

            It does **not** try to retell the full global-ML and ensemble pipeline here.
            """
        ),
        code(
            """
            overview_df = pd.DataFrame(
                {
                    'Metric': [
                        'Train rows',
                        'Test rows',
                        'Features',
                        'Codes',
                        'Train sub-codes',
                        'Test sub-codes',
                        'Horizons',
                        'Train ts range',
                        'Test ts range',
                    ],
                    'Value': [
                        f"{overview['train_rows']:,}",
                        f"{overview['test_rows']:,}",
                        overview['feature_count'],
                        overview['codes'],
                        overview['sub_codes_train'],
                        overview['sub_codes_test'],
                        ', '.join(map(str, overview['horizons'])),
                        f"{overview['train_ts_min']}–{overview['train_ts_max']}",
                        f"{overview['test_ts_min']}–{overview['test_ts_max']}",
                    ],
                }
            )
            display(overview_df)
            """
        ),
        md(
            """
            ## 2. Primary metric

            The project still uses the weighted skill score:

            \\[
            \\text{skill} = \\sqrt{1 - \\operatorname{clip}_{0,1}
            \\left(\\frac{\\sum_i w_i (y_i - \\hat y_i)^2}{\\sum_i w_i y_i^2}\\right)}
            \\]

            **Interpretation**
            - higher is better
            - `1.0` is perfect
            - weighting is central, not optional
            """
        ),
        md(
            """
            ## 3. Unified split policy

            The old notebook mismatch is removed here. Classical and deep notebooks will use the same per-series chronological split rule.

            **Rule**
            - first 80% of each selected series = training
            - last 20% = validation
            - minimum series length = 120
            - minimum validation length = 24
            """
        ),
        code(
            """
            split = overview['split_policy']

            fig, ax = plt.subplots(figsize=(10, 2.2))
            ax.barh(['Unified split'], [split['train_fraction']], color='#4C78A8', label='Train')
            ax.barh(['Unified split'], [1 - split['train_fraction']], left=[split['train_fraction']], color='#F58518', label='Validation')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Fraction of each selected series')
            ax.set_title('Shared chronological split used in 02_classical and 03_deep')
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
            fig.tight_layout()
            export_figure(fig, '00_unified_split_policy.png')
            plt.show()
            """
        ),
        md(
            """
            **Final deck figure:** `00_unified_split_policy.png`
            """
        ),
        md(
            """
            ## 4. Two evidence layers

            The rebuilt project uses:
            - **4 representatives** for high-clarity teaching
            - **1 fixed broader benchmark** so conclusions are not only anecdotal
            """
        ),
        code(
            """
            display(chosen[['series_id', 'reason', 'length', 'total_weight', 'target_std']])
            display(benchmark[['series_id', 'horizon', 'benchmark_band', 'length', 'total_weight', 'target_std']])
            """
        ),
        md(
            """
            ## 5. What we learned

            - the codex notebooks will teach a narrower but cleaner project story
            - split design is now unified before any model comparison
            - the old 4 anchor series are preserved as the main explanatory devices
            - the broader benchmark gives us a small reality check beyond those 4 series

            ## What changes next

            `01_eda.ipynb` will justify the model choices before any forecasting results are shown.
            """
        ),
    ]


def notebook_01():
    return [
        md(
            """
            # 01. EDA for the Rebuilt Notebook Story

            **By the end of this notebook you should understand**
            - the panel structure and scale
            - why weights dominate evaluation
            - why the old fixed cutoff was restrictive
            - what stationarity and lag structure imply for modeling
            """
        ),
        code(
            f"""
            {COMMON_SETUP}

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            from codex_notebooks.support import (
                LEGACY_VAL_CUTOFF,
                apply_plot_style,
                build_codex_artifacts,
                export_figure,
                load_artifact,
                load_json_artifact,
                load_study_series,
            )

            apply_plot_style()
            build_codex_artifacts(force=False, verbose=False)

            overview = load_json_artifact('overview.json')
            manifest = load_artifact('series_manifest.parquet')
            chosen = load_artifact('chosen_manifest.parquet')
            target_hist_full = load_artifact('target_hist_full.parquet')
            target_hist_clip = load_artifact('target_hist_clip.parquet')
            weight_hist_clip = load_artifact('weight_hist_clip.parquet')
            weight_cumshare = load_artifact('weight_cumshare.parquet')
            stationarity = load_artifact('stationarity_sample.parquet')
            lag_summary = load_artifact('lag_summary.parquet')
            study_series = load_study_series()
            representative_series = study_series[study_series['role'] == 'representative'].copy()
            """
        ),
        md(
            """
            ## 1. Panel structure

            The dataset is a large panel, not a single time series. That matters because identifiers, horizons, time order, and weights all matter simultaneously.
            """
        ),
        code(
            """
            structure_df = pd.DataFrame(
                {
                    'Measure': ['Total series', 'Codex-eligible series', 'Legacy-cutoff crossing', 'Legacy eligible (cross + length>=120)'],
                    'Value': [
                        f"{overview['legacy_cutoff_counts']['total_series']:,}",
                        f"{overview['codex_counts']['eligible_codex']:,}",
                        f"{overview['legacy_cutoff_counts']['crosses_legacy_cutoff']:,}",
                        f"{overview['legacy_cutoff_counts']['legacy_eligible']:,}",
                    ],
                }
            )
            display(structure_df)
            """
        ),
        md(
            """
            ## 2. Target distribution

            Heavy tails mean a model can look visually decent in the center of the distribution while still doing badly on error-heavy spikes.

            **Final deck figure:** `01_target_distribution.png`
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].bar(
                (target_hist_full['bin_left'] + target_hist_full['bin_right']) / 2,
                target_hist_full['count'],
                width=(target_hist_full['bin_right'] - target_hist_full['bin_left']),
                color='#C44E52',
                alpha=0.85,
            )
            axes[0].set_title('Raw target distribution')
            axes[0].set_xlabel('y_target')
            axes[0].set_ylabel('Count')

            axes[1].bar(
                (target_hist_clip['bin_left'] + target_hist_clip['bin_right']) / 2,
                target_hist_clip['count'],
                width=(target_hist_clip['bin_right'] - target_hist_clip['bin_left']),
                color='#8172B2',
                alpha=0.85,
            )
            axes[1].set_title('Target clipped to [1%, 99%]')
            axes[1].set_xlabel('y_target')
            axes[1].set_ylabel('Count')

            fig.tight_layout()
            export_figure(fig, '01_target_distribution.png')
            plt.show()
            """
        ),
        md(
            """
            ## 3. Weight concentration

            The weighted skill score is only interpretable if we know how concentrated the weights are.

            **Final deck figure:** `01_weight_concentration.png`
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].bar(
                (weight_hist_clip['bin_left'] + weight_hist_clip['bin_right']) / 2,
                weight_hist_clip['count'],
                width=(weight_hist_clip['bin_right'] - weight_hist_clip['bin_left']),
                color='#4C78A8',
                alpha=0.85,
            )
            axes[0].set_title('Weight distribution (clipped at 99th pct)')
            axes[0].set_xlabel('weight')
            axes[0].set_ylabel('Count')

            axes[1].plot(weight_cumshare['rank_pct'], weight_cumshare['cum_weight_share'], color='#F58518', linewidth=2)
            axes[1].set_title('Cumulative share of total weight')
            axes[1].set_xlabel('Rows ranked by weight (%)')
            axes[1].set_ylabel('Cumulative weight share')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1.01)

            fig.tight_layout()
            export_figure(fig, '01_weight_concentration.png')
            plt.show()

            pd.DataFrame(
                {
                    'Share': ['Top 1%', 'Top 5%', 'Top 10%'],
                    'Weight share': [
                        overview['weight_summary']['top_1pct_share'],
                        overview['weight_summary']['top_5pct_share'],
                        overview['weight_summary']['top_10pct_share'],
                    ],
                }
            )
            """
        ),
        md(
            """
            ## 4. Why the old cutoff was restrictive

            The rebuilt notebooks no longer use the legacy global cutoff for model comparison, but this EDA still explains why the old setup was restrictive.

            **Final deck figure:** `01_cutoff_feasibility.png`
            """
        ),
        code(
            """
            counts = pd.DataFrame(
                {
                    'Category': ['Total series', 'Cross legacy cutoff', 'Legacy eligible', 'Codex eligible'],
                    'Count': [
                        overview['legacy_cutoff_counts']['total_series'],
                        overview['legacy_cutoff_counts']['crosses_legacy_cutoff'],
                        overview['legacy_cutoff_counts']['legacy_eligible'],
                        overview['codex_counts']['eligible_codex'],
                    ],
                }
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(counts['Category'], counts['Count'], color=['#9ecae1', '#6baed6', '#3182bd', '#31a354'])
            ax.set_title('Series coverage under legacy cutoff vs codex eligibility')
            ax.set_ylabel('Number of series')
            ax.tick_params(axis='x', rotation=15)
            fig.tight_layout()
            export_figure(fig, '01_cutoff_feasibility.png')
            plt.show()
            counts
            """
        ),
        md(
            """
            ## 5. Representative anchor series

            The 4 anchor series are preserved because they teach different failure modes and strengths.

            **Final deck figure:** `01_representative_series.png`
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
            axes = axes.flatten()
            order = ['longest history', 'highest total weight', 'most volatile', 'most stable']
            chosen_ordered = chosen.set_index('reason').loc[order].reset_index()

            for ax, (_, row) in zip(axes, chosen_ordered.iterrows()):
                s = representative_series[representative_series['series_id'] == row['series_id']].sort_values('ts_index')
                ax.plot(s['ts_index'], s['y_target'], color='#4C78A8', linewidth=1.2)
                ax.set_title(f"{row['reason']}\\nH{int(row['horizon'])} | len={int(row['length'])}")
                ax.set_xlabel('ts_index')
                ax.set_ylabel('y_target')

            fig.tight_layout()
            export_figure(fig, '01_representative_series.png')
            plt.show()
            """
        ),
        md(
            """
            ## 6. Stationarity and differencing

            Raw levels are not universally safe. We compare stationarity verdicts before and after first differencing on a deterministic sample of eligible series.

            **Final deck figure:** `01_stationarity.png`
            """
        ),
        code(
            """
            verdict_order = ['stationary', 'mixed', 'unit_root', 'failed']
            verdict_counts = (
                stationarity[['verdict', 'verdict_d1']]
                .melt(var_name='phase', value_name='result')
                .assign(phase=lambda d: d['phase'].map({'verdict': 'raw level', 'verdict_d1': 'first difference'}))
                .groupby(['phase', 'result'])
                .size()
                .reset_index(name='count')
            )

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            raw_stationary = float((stationarity['adf_p'] < 0.05).mean())
            diff_stationary = float((stationarity['adf_p_d1'] < 0.05).mean())
            axes[0].bar(['Raw level', 'First difference'], [raw_stationary, diff_stationary], color=['#C44E52', '#55A868'])
            axes[0].set_ylim(0, 1.05)
            axes[0].set_ylabel('Share with ADF p < 0.05')
            axes[0].set_title('ADF stationary share')

            pivot = verdict_counts.pivot(index='result', columns='phase', values='count').reindex(verdict_order).fillna(0)
            pivot[['raw level', 'first difference']].plot.bar(ax=axes[1], color=['#8172B2', '#64B5CD'])
            axes[1].set_title('ADF + KPSS verdict counts')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=20)

            fig.tight_layout()
            export_figure(fig, '01_stationarity.png')
            plt.show()
            """
        ),
        md(
            """
            ## 7. Panel-level lag structure

            We aggregate lag structure over a deterministic sample rather than trusting a single anchor series.

            **Final deck figure:** `01_panel_lags.png`
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(lag_summary['lag'], lag_summary['mean_abs_acf'], color='#4C78A8')
            axes[0].set_title('Panel mean |ACF| by lag')
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('Mean absolute ACF')

            axes[1].bar(lag_summary['lag'], lag_summary['mean_abs_pacf'], color='#F58518')
            axes[1].set_title('Panel mean |PACF| by lag')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Mean absolute PACF')

            fig.tight_layout()
            export_figure(fig, '01_panel_lags.png')
            plt.show()
            """
        ),
        md(
            """
            ## 8. What we learned

            - the target is heavy-tailed and the weights are extremely concentrated
            - the old `VAL_CUTOFF = 2880` logic produced a narrow usable pool
            - first differencing materially improves stationarity evidence
            - short lags matter more than complex seasonal structure in this rebuilt notebook view

            ## What this changes next

            `02_classical.ipynb` will use the shared per-series chronological split and test simple baselines plus teachable classical models.
            """
        ),
    ]


def notebook_02():
    return [
        md(
            """
            # 02. Classical Forecasting Under a Shared Split Rule

            **By the end of this notebook you should understand**
            - how the codex classical study is evaluated
            - which simple models hold up on the 4 anchor series
            - how those models behave on a small broader benchmark
            """
        ),
        code(
            COMMON_SETUP
            + """

            import warnings
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from statsmodels.tsa.api import Holt, SimpleExpSmoothing
            from statsmodels.tsa.ar_model import AutoReg
            from statsmodels.tsa.arima.model import ARIMA

            from codex_notebooks.support import (
                CODEX_DIR,
                SERIES_KEYS,
                apply_plot_style,
                build_codex_artifacts,
                chronological_split_df,
                export_figure,
                load_artifact,
                load_study_series,
                mase,
                weighted_mae,
                weighted_rmse,
                weighted_skill,
            )

            warnings.filterwarnings('ignore')
            apply_plot_style()
            build_codex_artifacts(force=False, verbose=False)

            chosen = load_artifact('chosen_manifest.parquet')
            benchmark = load_artifact('benchmark_manifest.parquet')
            study_series = load_study_series()
            series_frames = {
                series_id: frame.sort_values('ts_index').reset_index(drop=True)
                for series_id, frame in study_series.groupby('series_id')
            }

            RESULTS_PATH = CODEX_DIR / 'classical_results.parquet'
            FORECASTS_PATH = CODEX_DIR / 'classical_forecasts.parquet'
            """
        ),
        md(
            """
            ## 1. Model lineup

            This notebook keeps the lineup teachable:
            - `Naive`
            - `Drift`
            - `SES`
            - `Holt`
            - `AR(1)`
            - `ARIMA(1,1,1)`
            """
        ),
        code(
            """
            MODEL_META = pd.DataFrame(
                [
                    {'model': 'Naive', 'family': 'baseline'},
                    {'model': 'Drift', 'family': 'baseline'},
                    {'model': 'SES', 'family': 'smoothing'},
                    {'model': 'Holt', 'family': 'smoothing'},
                    {'model': 'AR(1)', 'family': 'autoregressive'},
                    {'model': 'ARIMA(1,1,1)', 'family': 'arima'},
                ]
            )
            MODEL_META
            """
        ),
        code(
            """
            def forecast_classical(model_name, train_values, steps):
                train_values = np.asarray(train_values, dtype=float)
                if model_name == 'Naive':
                    return np.repeat(train_values[-1], steps)
                if model_name == 'Drift':
                    if len(train_values) < 2:
                        return np.repeat(train_values[-1], steps)
                    slope = (train_values[-1] - train_values[0]) / (len(train_values) - 1)
                    return train_values[-1] + slope * np.arange(1, steps + 1)
                if model_name == 'SES':
                    fit = SimpleExpSmoothing(train_values, initialization_method='estimated').fit(optimized=True)
                    return np.asarray(fit.forecast(steps), dtype=float)
                if model_name == 'Holt':
                    fit = Holt(train_values, initialization_method='estimated').fit(optimized=True)
                    return np.asarray(fit.forecast(steps), dtype=float)
                if model_name == 'AR(1)':
                    fit = AutoReg(train_values, lags=1, old_names=False).fit()
                    return np.asarray(fit.predict(start=len(train_values), end=len(train_values) + steps - 1), dtype=float)
                if model_name == 'ARIMA(1,1,1)':
                    fit = ARIMA(train_values, order=(1, 1, 1)).fit()
                    return np.asarray(fit.forecast(steps), dtype=float)
                raise KeyError(model_name)


            def evaluate_series(series_id, frame, role, label):
                train_part, val_part, info = chronological_split_df(frame[['ts_index', 'y_target', 'weight']])
                rows = []
                forecast_rows = []
                for model_name in MODEL_META['model']:
                    try:
                        pred = forecast_classical(model_name, train_part['y_target'].to_numpy(), len(val_part))
                        status = 'ok'
                    except Exception as exc:
                        pred = np.repeat(np.nan, len(val_part))
                        status = f'error: {type(exc).__name__}'
                    rows.append(
                        {
                            'series_id': series_id,
                            'role': role,
                            'label': label,
                            'model': model_name,
                            'train_len': info.train_len,
                            'val_len': info.val_len,
                            'status': status,
                            'skill_score': weighted_skill(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'rmse': weighted_rmse(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'mae': weighted_mae(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'mase': mase(val_part['y_target'], pred, train_part['y_target']) if status == 'ok' else np.nan,
                        }
                    )
                    for ts_index, y_true, y_pred in zip(val_part['ts_index'], val_part['y_target'], pred):
                        forecast_rows.append(
                            {
                                'series_id': series_id,
                                'role': role,
                                'label': label,
                                'model': model_name,
                                'ts_index': ts_index,
                                'y_true': y_true,
                                'y_pred': y_pred,
                            }
                        )
                return rows, forecast_rows
            """
        ),
        code(
            """
            if RESULTS_PATH.exists() and FORECASTS_PATH.exists():
                results = pd.read_parquet(RESULTS_PATH)
                forecasts = pd.read_parquet(FORECASTS_PATH)
            else:
                result_rows = []
                forecast_rows = []

                reps = chosen[['series_id', 'reason']].copy().rename(columns={'reason': 'label'})
                bench = benchmark[['series_id', 'benchmark_band']].copy().rename(columns={'benchmark_band': 'label'})

                for _, row in reps.iterrows():
                    r_rows, f_rows = evaluate_series(row['series_id'], series_frames[row['series_id']], 'representative', row['label'])
                    result_rows.extend(r_rows)
                    forecast_rows.extend(f_rows)

                for _, row in bench.iterrows():
                    r_rows, f_rows = evaluate_series(row['series_id'], series_frames[row['series_id']], 'benchmark', row['label'])
                    result_rows.extend(r_rows)
                    forecast_rows.extend(f_rows)

                results = pd.DataFrame(result_rows)
                forecasts = pd.DataFrame(forecast_rows)
                results.to_parquet(RESULTS_PATH, index=False)
                forecasts.to_parquet(FORECASTS_PATH, index=False)

            ok_results = results[results['status'] == 'ok'].copy()
            best_by_series = ok_results.sort_values(['series_id', 'skill_score'], ascending=[True, False]).groupby('series_id').head(1)
            """
        ),
        md(
            """
            ## 2. Representative-series results

            These are the 4 main teaching cases.
            """
        ),
        code(
            """
            rep_summary = ok_results[ok_results['role'] == 'representative'].pivot_table(
                index='label',
                columns='model',
                values='skill_score'
            ).round(3)
            display(rep_summary)
            """
        ),
        md(
            """
            **Final deck figure:** `02_classical_forecasts.png`
            """
        ),
        code(
            """
            rep_best = best_by_series[best_by_series['role'] == 'representative'].copy()
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
            axes = axes.flatten()

            for ax, (_, row) in zip(axes, rep_best.iterrows()):
                full_series = series_frames[row['series_id']].sort_values('ts_index').reset_index(drop=True)
                train_part, val_part, info = chronological_split_df(full_series[['ts_index', 'y_target', 'weight']])
                naive_pred = forecasts[(forecasts['series_id'] == row['series_id']) & (forecasts['model'] == 'Naive')]
                best_pred = forecasts[(forecasts['series_id'] == row['series_id']) & (forecasts['model'] == row['model'])]

                ax.plot(full_series['ts_index'], full_series['y_target'], color='#bdbdbd', linewidth=1.0, label='full series')
                ax.axvline(train_part['ts_index'].iloc[-1], color='black', linestyle='--', linewidth=1, label='split')
                ax.plot(val_part['ts_index'], val_part['y_target'], color='#222222', linewidth=1.6, label='validation truth')
                ax.plot(naive_pred['ts_index'], naive_pred['y_pred'], color='#F58518', linestyle=':', linewidth=1.5, label='naive baseline')
                ax.plot(best_pred['ts_index'], best_pred['y_pred'], color='#4C78A8', linewidth=1.8, label='best model forecast')
                ax.set_title(f"{row['label']} | best={row['model']} | skill={row['skill_score']:.3f}")
                ax.set_xlabel('ts_index')
                ax.set_ylabel('y_target')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout()
            export_figure(fig, '02_classical_forecasts.png')
            plt.show()
            """
        ),
        md(
            """
            ## 3. Broader benchmark

            The broader benchmark is small on purpose, but it tests whether the representative story is too anecdotal.

            **Final deck figure:** `02_classical_benchmark_skill.png`
            """
        ),
        code(
            """
            benchmark_summary = (
                ok_results[ok_results['role'] == 'benchmark']
                .groupby('model')
                .agg(
                    mean_skill=('skill_score', 'mean'),
                    median_skill=('skill_score', 'median'),
                    mean_rmse=('rmse', 'mean'),
                )
                .sort_values('mean_skill', ascending=False)
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(benchmark_summary['model'], benchmark_summary['mean_skill'], color='#4C78A8')
            ax.set_title('Classical mean weighted skill on the broader benchmark')
            ax.set_ylabel('Mean weighted skill')
            ax.tick_params(axis='x', rotation=20)
            fig.tight_layout()
            export_figure(fig, '02_classical_benchmark_skill.png')
            plt.show()

            benchmark_summary
            """
        ),
        md(
            """
            ## 4. What we learned

            - the shared split rule makes the classical study directly comparable to the deep study
            - simple baselines still matter
            - some series reward smoothing, while others need a more structured model
            - the small broader benchmark gives us a more stable reading than 4 examples alone

            ## What this changes next

            `03_deep.ipynb` will keep the same split rule and benchmark, then test target-history recurrent models on the same study set.
            """
        ),
    ]


def notebook_03():
    return [
        md(
            """
            # 03. Deep Sequence Models Under the Same Split Rule

            **By the end of this notebook you should understand**
            - how the codex deep study is trained and evaluated
            - where RNN / GRU / LSTM help on the representative cases
            - how deep models compare to classical ones on the shared benchmark
            """
        ),
        code(
            COMMON_SETUP
            + """

            import copy
            import warnings

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import display
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            from codex_notebooks.support import (
                CODEX_DIR,
                LOOKBACK,
                apply_plot_style,
                build_codex_artifacts,
                chronological_split_df,
                export_figure,
                load_artifact,
                load_study_series,
                mase,
                weighted_mae,
                weighted_rmse,
                weighted_skill,
            )

            warnings.filterwarnings('ignore')
            apply_plot_style()
            build_codex_artifacts(force=False, verbose=False)

            torch.manual_seed(42)
            np.random.seed(42)
            DEVICE = torch.device('cpu')
            MAX_EPOCHS = 20
            PATIENCE = 4
            HIDDEN_SIZE = 24
            BATCH_SIZE = 32
            LR = 1e-3

            chosen = load_artifact('chosen_manifest.parquet')
            benchmark = load_artifact('benchmark_manifest.parquet')
            study_series = load_study_series()
            series_frames = {
                series_id: frame.sort_values('ts_index').reset_index(drop=True)
                for series_id, frame in study_series.groupby('series_id')
            }

            RESULTS_PATH = CODEX_DIR / 'deep_results.parquet'
            FORECASTS_PATH = CODEX_DIR / 'deep_forecasts.parquet'
            """
        ),
        md(
            """
            ## 1. Model setup

            This is intentionally modest and class-teachable:
            - target history only
            - one-step training windows
            - recursive multi-step validation forecasts
            - `RNN`, `GRU`, and `LSTM`
            """
        ),
        code(
            """
            MODEL_NAMES = ['Naive', 'RNN', 'GRU', 'LSTM']


            def make_windows(values, lookback=LOOKBACK):
                values = np.asarray(values, dtype=np.float32)
                X, y = [], []
                for i in range(len(values) - lookback):
                    X.append(values[i:i + lookback])
                    y.append(values[i + lookback])
                return np.asarray(X), np.asarray(y)


            class RecurrentRegressor(nn.Module):
                def __init__(self, kind, hidden_size=HIDDEN_SIZE):
                    super().__init__()
                    if kind == 'RNN':
                        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
                    elif kind == 'GRU':
                        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
                    elif kind == 'LSTM':
                        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                    else:
                        raise KeyError(kind)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    out, _ = self.rnn(x)
                    return self.fc(out[:, -1, :]).squeeze(-1)


            def fit_recurrent(kind, train_values):
                mean = float(np.mean(train_values))
                std = float(np.std(train_values))
                if std < 1e-8:
                    std = 1.0

                scaled = (np.asarray(train_values, dtype=np.float32) - mean) / std
                X, y = make_windows(scaled)
                if len(X) < 8:
                    raise ValueError('not enough windows')

                split = max(1, int(len(X) * 0.8))
                train_X = torch.tensor(X[:split]).unsqueeze(-1)
                train_y = torch.tensor(y[:split])
                val_X = torch.tensor(X[split:]).unsqueeze(-1) if split < len(X) else train_X[-1:].clone()
                val_y = torch.tensor(y[split:]) if split < len(X) else train_y[-1:].clone()

                loader = DataLoader(TensorDataset(train_X, train_y), batch_size=min(BATCH_SIZE, len(train_X)), shuffle=True)
                model = RecurrentRegressor(kind).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                loss_fn = nn.MSELoss()

                best_state = None
                best_val = float('inf')
                wait = 0

                for epoch in range(MAX_EPOCHS):
                    model.train()
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_pred = model(val_X.to(DEVICE))
                        val_loss = float(loss_fn(val_pred, val_y.to(DEVICE)).item())

                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        wait = 0
                    else:
                        wait += 1
                        if wait >= PATIENCE:
                            break

                if best_state is not None:
                    model.load_state_dict(best_state)
                return model, mean, std


            def recursive_forecast(model, train_values, steps, mean, std):
                context = list(((np.asarray(train_values, dtype=np.float32) - mean) / std).tolist())
                preds = []
                model.eval()
                with torch.no_grad():
                    for _ in range(steps):
                        x = torch.tensor(context[-LOOKBACK:], dtype=torch.float32).view(1, LOOKBACK, 1).to(DEVICE)
                        pred_scaled = float(model(x).cpu().item())
                        preds.append(pred_scaled * std + mean)
                        context.append(pred_scaled)
                return np.asarray(preds, dtype=float)
            """
        ),
        code(
            """
            def evaluate_deep_series(series_id, frame, role, label):
                train_part, val_part, info = chronological_split_df(frame[['ts_index', 'y_target', 'weight']])
                rows = []
                forecast_rows = []

                naive_pred = np.repeat(train_part['y_target'].iloc[-1], len(val_part))
                rows.append(
                    {
                        'series_id': series_id,
                        'role': role,
                        'label': label,
                        'model': 'Naive',
                        'train_len': info.train_len,
                        'val_len': info.val_len,
                        'status': 'ok',
                        'skill_score': weighted_skill(val_part['y_target'], naive_pred, val_part['weight']),
                        'rmse': weighted_rmse(val_part['y_target'], naive_pred, val_part['weight']),
                        'mae': weighted_mae(val_part['y_target'], naive_pred, val_part['weight']),
                        'mase': mase(val_part['y_target'], naive_pred, train_part['y_target']),
                    }
                )
                for ts_index, y_true, y_pred in zip(val_part['ts_index'], val_part['y_target'], naive_pred):
                    forecast_rows.append({'series_id': series_id, 'role': role, 'label': label, 'model': 'Naive', 'ts_index': ts_index, 'y_true': y_true, 'y_pred': y_pred})

                for kind in ['RNN', 'GRU', 'LSTM']:
                    try:
                        model, mean, std = fit_recurrent(kind, train_part['y_target'].to_numpy())
                        pred = recursive_forecast(model, train_part['y_target'].to_numpy(), len(val_part), mean, std)
                        status = 'ok'
                    except Exception as exc:
                        pred = np.repeat(np.nan, len(val_part))
                        status = f'error: {type(exc).__name__}'

                    rows.append(
                        {
                            'series_id': series_id,
                            'role': role,
                            'label': label,
                            'model': kind,
                            'train_len': info.train_len,
                            'val_len': info.val_len,
                            'status': status,
                            'skill_score': weighted_skill(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'rmse': weighted_rmse(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'mae': weighted_mae(val_part['y_target'], pred, val_part['weight']) if status == 'ok' else np.nan,
                            'mase': mase(val_part['y_target'], pred, train_part['y_target']) if status == 'ok' else np.nan,
                        }
                    )
                    for ts_index, y_true, y_pred in zip(val_part['ts_index'], val_part['y_target'], pred):
                        forecast_rows.append({'series_id': series_id, 'role': role, 'label': label, 'model': kind, 'ts_index': ts_index, 'y_true': y_true, 'y_pred': y_pred})
                return rows, forecast_rows
            """
        ),
        code(
            """
            if RESULTS_PATH.exists() and FORECASTS_PATH.exists():
                results = pd.read_parquet(RESULTS_PATH)
                forecasts = pd.read_parquet(FORECASTS_PATH)
            else:
                result_rows = []
                forecast_rows = []

                reps = chosen[['series_id', 'reason']].copy().rename(columns={'reason': 'label'})
                bench = benchmark[['series_id', 'benchmark_band']].copy().rename(columns={'benchmark_band': 'label'})

                for _, row in reps.iterrows():
                    r_rows, f_rows = evaluate_deep_series(row['series_id'], series_frames[row['series_id']], 'representative', row['label'])
                    result_rows.extend(r_rows)
                    forecast_rows.extend(f_rows)

                for _, row in bench.iterrows():
                    r_rows, f_rows = evaluate_deep_series(row['series_id'], series_frames[row['series_id']], 'benchmark', row['label'])
                    result_rows.extend(r_rows)
                    forecast_rows.extend(f_rows)

                results = pd.DataFrame(result_rows)
                forecasts = pd.DataFrame(forecast_rows)
                results.to_parquet(RESULTS_PATH, index=False)
                forecasts.to_parquet(FORECASTS_PATH, index=False)

            ok_results = results[results['status'] == 'ok'].copy()
            best_by_series = ok_results.sort_values(['series_id', 'skill_score'], ascending=[True, False]).groupby('series_id').head(1)
            """
        ),
        md(
            """
            ## 2. Representative-series results
            """
        ),
        code(
            """
            rep_summary = ok_results[ok_results['role'] == 'representative'].pivot_table(
                index='label',
                columns='model',
                values='skill_score'
            ).round(3)
            display(rep_summary)
            """
        ),
        md(
            """
            **Final deck figure:** `03_deep_forecasts.png`
            """
        ),
        code(
            """
            rep_best = best_by_series[best_by_series['role'] == 'representative'].copy()
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
            axes = axes.flatten()

            for ax, (_, row) in zip(axes, rep_best.iterrows()):
                full_series = series_frames[row['series_id']].sort_values('ts_index').reset_index(drop=True)
                train_part, val_part, info = chronological_split_df(full_series[['ts_index', 'y_target', 'weight']])
                naive_pred = forecasts[(forecasts['series_id'] == row['series_id']) & (forecasts['model'] == 'Naive')]
                best_pred = forecasts[(forecasts['series_id'] == row['series_id']) & (forecasts['model'] == row['model'])]

                ax.plot(full_series['ts_index'], full_series['y_target'], color='#bdbdbd', linewidth=1.0, label='full series')
                ax.axvline(train_part['ts_index'].iloc[-1], color='black', linestyle='--', linewidth=1, label='split')
                ax.plot(val_part['ts_index'], val_part['y_target'], color='#222222', linewidth=1.6, label='validation truth')
                ax.plot(naive_pred['ts_index'], naive_pred['y_pred'], color='#F58518', linestyle=':', linewidth=1.5, label='naive baseline')
                ax.plot(best_pred['ts_index'], best_pred['y_pred'], color='#4C78A8', linewidth=1.8, label='best model forecast')
                ax.set_title(f"{row['label']} | best={row['model']} | skill={row['skill_score']:.3f}")
                ax.set_xlabel('ts_index')
                ax.set_ylabel('y_target')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout()
            export_figure(fig, '03_deep_forecasts.png')
            plt.show()
            """
        ),
        md(
            """
            ## 3. Broader benchmark

            **Final deck figure:** `03_deep_benchmark_skill.png`
            """
        ),
        code(
            """
            benchmark_summary = (
                ok_results[ok_results['role'] == 'benchmark']
                .groupby('model')
                .agg(
                    mean_skill=('skill_score', 'mean'),
                    median_skill=('skill_score', 'median'),
                    mean_rmse=('rmse', 'mean'),
                )
                .sort_values('mean_skill', ascending=False)
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(benchmark_summary['model'], benchmark_summary['mean_skill'], color='#55A868')
            ax.set_title('Deep mean weighted skill on the broader benchmark')
            ax.set_ylabel('Mean weighted skill')
            ax.tick_params(axis='x', rotation=20)
            fig.tight_layout()
            export_figure(fig, '03_deep_benchmark_skill.png')
            plt.show()

            benchmark_summary
            """
        ),
        md(
            """
            ## 4. Shared-benchmark comparison to classical

            Because `02_classical` and `03_deep` use the same split rule and the same benchmark series, we can compare them directly here.

            **Final deck figure:** `03_classical_vs_deep_benchmark.png`
            """
        ),
        code(
            """
            classical_results = pd.read_parquet(CODEX_DIR / 'classical_results.parquet')
            classical_ok = classical_results[classical_results['status'] == 'ok']

            classical_best = classical_ok[classical_ok['role'] == 'benchmark'].sort_values(['series_id', 'skill_score'], ascending=[True, False]).groupby('series_id').head(1)
            deep_best = ok_results[ok_results['role'] == 'benchmark'].sort_values(['series_id', 'skill_score'], ascending=[True, False]).groupby('series_id').head(1)

            compare = (
                benchmark[['series_id', 'horizon']]
                .merge(classical_best[['series_id', 'skill_score']].rename(columns={'skill_score': 'best_classical'}), on='series_id', how='left')
                .merge(deep_best[['series_id', 'skill_score']].rename(columns={'skill_score': 'best_deep'}), on='series_id', how='left')
            )

            compare_summary = compare.groupby('horizon')[['best_classical', 'best_deep']].mean().reset_index()

            x = np.arange(len(compare_summary))
            width = 0.36
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x - width / 2, compare_summary['best_classical'], width=width, color='#4C78A8', label='Best classical')
            ax.bar(x + width / 2, compare_summary['best_deep'], width=width, color='#55A868', label='Best deep')
            ax.set_xticks(x)
            ax.set_xticklabels([f"H{int(h)}" for h in compare_summary['horizon']])
            ax.set_ylabel('Mean best-model weighted skill')
            ax.set_title('Best classical vs best deep on the shared benchmark')
            ax.legend()
            fig.tight_layout()
            export_figure(fig, '03_classical_vs_deep_benchmark.png')
            plt.show()

            compare_summary
            """
        ),
        md(
            """
            ## 5. What we learned

            - the deep notebook now shares the same split rule as the classical notebook
            - target-history recurrent models can be compared honestly against classical models on the shared benchmark
            - the representative cases still matter for intuition, but the benchmark prevents over-claiming

            ## What changes next

            The codex notebook phase is complete when these notebooks are runnable, explain the project clearly, and export the figures needed for the Beamer deck.
            """
        ),
    ]


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nbf.write(nb, path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "codex_notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_notebook(out_dir / "00_project_map.ipynb", notebook_00())
    write_notebook(out_dir / "01_eda.ipynb", notebook_01())
    write_notebook(out_dir / "02_classical.ipynb", notebook_02())
    write_notebook(out_dir / "03_deep.ipynb", notebook_03())

    readme = dedent(
        """
        # Codex Notebooks

        This folder contains the rebuilt teaching-first notebook layer for the project.

        ## Run order
        1. `00_project_map.ipynb`
        2. `01_eda.ipynb`
        3. `02_classical.ipynb`
        4. `03_deep.ipynb`

        The notebooks rely on processed artifacts in `data/processed/codex/`.
        If those artifacts are missing, the notebooks build them on first run.
        """
    ).strip() + "\n"
    (out_dir / "README.md").write_text(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
