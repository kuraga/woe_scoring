import os
from functools import lru_cache, partial
from itertools import combinations
from typing import Dict, List, Union, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .model import Model


class GiniScoreCalculator:
    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_score(data: Union[pd.DataFrame, np.ndarray],
                       target: Union[pd.Series, np.ndarray],
                       feature: str,
                       random_state: int,
                       class_weight: str,
                       cv: int,
                       scoring: str,
                       n_jobs: int) -> float:
        """Calculate the Gini score for a given feature using Logistic Regression."""

        model = LogisticRegression(random_state=random_state,
                                 class_weight=class_weight,
                                 max_iter=1000,
                                 n_jobs=n_jobs,
                                 warm_start=True)

        X = data[feature].values.reshape(-1, 1)
        scores = cross_val_score(estimator=model,
                               X=X,
                               y=target,
                               cv=cv,
                               scoring=scoring,
                               n_jobs=n_jobs)

        return (np.mean(scores) * 2 - 1) * 100


class FeatureAnalyzer:
    def __init__(self):
        self.gini_calculator = GiniScoreCalculator()

    def calc_features_gini_quality(self,
                                 data: Union[pd.DataFrame, np.ndarray],
                                 target: Union[pd.Series, np.ndarray],
                                 feature_names: List[str],
                                 random_state: int,
                                 class_weight: str,
                                 cv: int,
                                 scoring: str,
                                 n_jobs: int) -> Dict[str, float]:
        """Calculate Gini quality scores for multiple features"""

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            calc_score = partial(self.gini_calculator.calculate_score,
                               data=data,
                               target=target,
                               random_state=random_state,
                               class_weight=class_weight,
                               cv=cv,
                               scoring=scoring,
                               n_jobs=1)

            scores = list(executor.map(calc_score, feature_names))
            return dict(zip(feature_names, scores))

    def check_gini_threshold(self,
                           feature_names: List[str],
                           gini_scores: Dict[str, float],
                           threshold: float) -> List[str]:
        """Get features above Gini threshold"""
        return [f for f in feature_names if gini_scores[f] >= threshold]

    def check_correlation(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         feature_names: List[str],
                         gini_scores: Dict[str, float],
                         threshold: float) -> List[str]:
        """Get uncorrelated features"""
        corr_matrix = data[feature_names].corr().abs()
        uncorrelated = set(feature_names)

        # Use numpy operations for faster correlation checks
        idx = np.triu_indices(len(feature_names), k=1)
        high_corr = np.where(corr_matrix.values[idx] >= threshold)[0]

        for i in high_corr:
            f1, f2 = feature_names[idx[0][i]], feature_names[idx[1][i]]
            f_remove = f2 if gini_scores[f1] > gini_scores[f2] else f1
            uncorrelated.discard(f_remove)

        return list(uncorrelated)

    def check_min_group_pct(self,
                           data: Union[pd.DataFrame, np.ndarray],
                           feature_names: List[str],
                           min_pct: float) -> List[str]:
        """Get features meeting minimum group percentage"""
        value_counts = {f: data[f].value_counts(normalize=True) for f in feature_names}
        to_drop = {f for f, vc in value_counts.items() if vc.min() < min_pct}
        return list(set(feature_names) - to_drop)


class ModelAnalyzer:
    @staticmethod
    def find_bad_features(model: Model) -> List[str]:
        """Find features with high p-values or positive coefficients"""
        mask = (model.pvalues_ > 0.05) | (model.coef_ > 0)
        return [f for i, f in enumerate(model.feature_names_) if mask[i]]

    @staticmethod
    def calc_iv_dict(data: pd.DataFrame,
                     target: np.ndarray,
                     feature: str) -> Dict:
        """Calculate information value for categorical feature"""
        crosstab = pd.crosstab(data[feature], target)
        total_bad, total_good = target.sum(), len(target) - target.sum()
        woe = np.log((crosstab[1] / total_bad) / (crosstab[0] / total_good))
        iv = ((crosstab[1] / total_bad) - (crosstab[0] / total_good)) * woe
        return {feature: iv.sum()}

    @staticmethod
    def save_reports(model: sm.Logit,
                    path: str = os.getcwd()) -> None:
        """Save model summary reports"""
        try:
            summary_path = os.path.join(path, "model_summary.txt")
            wald_path = os.path.join(path, "model_wald.txt")

            with open(summary_path, "w") as f:
                f.write(model.summary().as_text())

            with open(wald_path, "w") as f:
                model.wald_test_terms().summary_frame().to_string(f)

        except Exception as e:
            print(f"Problem saving reports: {e}")


class SQLGenerator:
    @staticmethod
    def generate(encoder: Any,
                feature_names: List[str],
                coef: List[float],
                intercept: float) -> str:
        """Generate SQL for model scoring"""
        sql_parts = ["with a as (SELECT "]
        sql_parts.append(",".join(var.replace("WOE_", "") for var in feature_names))

        for var in feature_names:
            clean_var = var.replace("WOE_", "")
            for woe_dict in encoder.woe_iv_dict:
                if list(woe_dict.keys())[0] == clean_var:
                    sql_parts.append(", CASE")

                    if woe_dict["type_feature"] == "cat":
                        sql_parts.extend(SQLGenerator._categorical_case(var, woe_dict))
                    else:
                        sql_parts.extend(SQLGenerator._numeric_case(var, woe_dict))

                    sql_parts.append(f" END AS {var}")
                    break

        sql_parts.extend(SQLGenerator._finish_sql(feature_names, coef, intercept))
        return "".join(sql_parts)

    @staticmethod
    def _categorical_case(var: str,
                         woe_dict: Dict) -> List[str]:
        """Generate categorical CASE statements"""
        feature = var.replace("WOE_", "")
        cases = []

        for bin_info in woe_dict[feature]:
            bin_str = str(bin_info['bin']).replace("[", "(").replace("]", ")")
            bin_str = bin_str.replace(", -1", "").replace(", Missing", "")
            cases.append(f" WHEN {feature} in {bin_str} THEN {bin_info['woe']}")

        if woe_dict["missing_bin"] == "first":
            cases.append(f" WHEN {feature} IS NULL THEN {woe_dict[feature][0]['woe']}")
            cases.append(f" ELSE {woe_dict[feature][0]['woe']}")
        elif woe_dict["missing_bin"] == "last":
            last_idx = len(woe_dict[feature]) - 1
            cases.append(f" WHEN {feature} IS NULL THEN {woe_dict[feature][last_idx]['woe']}")
            cases.append(f" ELSE {woe_dict[feature][last_idx]['woe']}")

        return cases

    @staticmethod
    def _numeric_case(var: str,
                     woe_dict: Dict) -> List[str]:
        """Generate numeric CASE statements"""
        feature = var.replace("WOE_", "")
        cases = []

        if woe_dict["missing_bin"] == "first":
            cases.append(f" WHEN {feature} IS NULL THEN {woe_dict[feature][0]['woe']}")
        elif woe_dict["missing_bin"] == "last":
            last_idx = len(woe_dict[feature]) - 1
            cases.append(f" WHEN {feature} IS NULL THEN {woe_dict[feature][last_idx]['woe']}")

        for i, bin_info in enumerate(woe_dict[feature]):
            if i == 0:
                cases.append(f" WHEN {feature} < {bin_info['bin'][1]} THEN {bin_info['woe']}")
            elif i == len(woe_dict[feature]) - 1:
                cases.append(f" WHEN {feature} >= {bin_info['bin'][0]} THEN {bin_info['woe']}")
            else:
                cases.append(
                    f" WHEN {feature} >= {bin_info['bin'][0]} AND {feature} < {bin_info['bin'][1]} THEN {bin_info['woe']}"
                )

        return cases

    @staticmethod
    def _finish_sql(feature_names: List[str],
                   coef: List[float],
                   intercept: float) -> List[str]:
        """Generate final SQL parts"""
        sql = [" FROM )",
               ", b as (",
               "SELECT a.*",
               f", REPLACE(1 / (1 + EXP(-({intercept}"]

        for idx, feature in enumerate(feature_names):
            sql.append(f" + ({coef[idx]} * a.{feature})")

        sql.extend(["))), ',', '.') as PD",
                   " FROM a) ",
                   "SELECT * FROM b"])
        return sql


def _calc_score_points(woe: float,
                      coef: float,
                      intercept: float,
                      factor: float,
                      offset: float,
                      n_features: int) -> float:
    """Calculate scorecard points"""
    return -(woe * coef + intercept / n_features) * factor + offset / n_features


class ScorecardBuilder:
    @staticmethod
    def build_scorecard(feature_names: List[str],
                       encoder: Any,
                       model_results: pd.DataFrame,
                       factor: float,
                       offset: float) -> List[pd.DataFrame]:
        """Build scorecard stats"""
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, feature in enumerate(model_results.iloc[:, 0]):
                futures.append(
                    executor.submit(
                        ScorecardBuilder._calc_feature_stats,
                        idx, feature, feature_names, encoder,
                        model_results, factor, offset
                    )
                )
            return [f.result() for f in futures]

    @staticmethod
    def _calc_feature_stats(idx: int,
                          feature: str,
                          feature_names: List[str],
                          encoder: Any,
                          model_results: pd.DataFrame,
                          factor: float,
                          offset: float) -> pd.DataFrame:
        """Calculate feature statistics"""
        result = {
            "feature": [],
            "coef": [],
            "pvalue": [],
            "bin": [],
            "WOE": [],
            "IV": [],
            "percent_of_population": [],
            "total": [],
            "event_cnt": [],
            "non_event_cnt": [],
            "event_rate": [],
            "score_ball": [],
        }

        ScorecardBuilder._update_base_stats(result, feature, model_results, idx)

        intercept = model_results.iloc[0, 1]
        n_features = len(feature_names)

        if idx < 1:
            for key in result:
                if key not in ["feature", "coef", "pvalue"]:
                    result[key].append("-")
        else:
            clean_feature = feature.replace("WOE_", "")
            for woe_dict in encoder.woe_iv_dict:
                if list(woe_dict.keys())[0] == clean_feature:
                    ScorecardBuilder._update_feature_stats(
                        result, woe_dict[clean_feature],
                        result["coef"][-1], intercept, factor, offset, n_features
                    )

        return pd.DataFrame.from_dict(result)

    @staticmethod
    def _update_base_stats(result: Dict,
                          feature: str,
                          model_results: pd.DataFrame,
                          idx: int) -> None:
        """Update basic feature stats"""
        result["feature"].append(feature.replace("WOE_", ""))
        result["coef"].append(model_results.loc[idx, "coef"])
        result["pvalue"].append(model_results.loc[idx, "P>|z|"])

    @staticmethod
    def _update_feature_stats(result: Dict,
                            feature_woe: Dict,
                            coef: float,
                            intercept: float,
                            factor: float,
                            offset: float,
                            n_features: int) -> None:
        """Update detailed feature stats"""
        for bin_info in feature_woe:
            bin_values = bin_info["bin"]
            bin_values_str = [str(v).replace("-1", "missing") if v == -1 else v
                            for v in bin_values]

            result["bin"].append(bin_values_str)
            result["WOE"].append(bin_info["woe"])
            result["IV"].append(bin_info["iv"])
            result["percent_of_population"].append(bin_info["pct"])
            result["total"].append(bin_info["total"])
            result["event_cnt"].append(bin_info["bad"])
            result["non_event_cnt"].append(bin_info["total"] - bin_info["bad"])
            result["event_rate"].append(bin_info["bad_rate"])
            result["score_ball"].append(
                _calc_score_points(
                    woe=bin_info["woe"],
                    coef=coef,
                    intercept=intercept,
                    factor=factor,
                    offset=offset,
                    n_features=n_features
                )
            )


class ExcelBuilder:
    @staticmethod
    def build_excel_sheet(feature_stats: List[pd.DataFrame],
                         writer: pd.ExcelWriter,
                         width: int = 640,
                         height: int = 480,
                         first_plot_pos: str = 'A',
                         second_plot_pos: str = 'J') -> None:
        """Build Excel scorecard with charts"""
        workbook = writer.book
        merge_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })

        const_results = [r for r in feature_stats if r['feature'][0] == 'const']
        feature_results = [r for r in feature_stats
                         if r is not None and r['feature'][0] != 'const']
        all_results = const_results + feature_results

        indexes = np.cumsum([len(r) for r in all_results])
        full_features = pd.concat(all_results, ignore_index=True)
        full_features.to_excel(writer, sheet_name='Scorecard')

        scorecard_sheet = writer.sheets['Scorecard']
        ExcelBuilder._format_scorecard(scorecard_sheet, all_results, indexes, merge_format)
        ExcelBuilder._add_feature_charts(feature_results, writer, workbook, width,
                                       height, first_plot_pos, second_plot_pos)

    @staticmethod
    def _format_scorecard(sheet,
                         results: List[pd.DataFrame],
                         indexes: List[int],
                         format) -> None:
        """Format the main scorecard sheet"""
        area_start = 1
        for result, index in zip(results, indexes):
            for col, width in zip([1, 2, 3], [20, 10, 10]):
                sheet.merge_range(area_start, col, index, col,
                                result.iloc[0, col-1], format)
                sheet.set_column(col, col, width)
            area_start = index + 1

    @staticmethod
    def _add_feature_charts(results: List[pd.DataFrame],
                          writer,
                          workbook,
                          width: int,
                          height: int,
                          first_pos: str,
                          second_pos: str) -> None:
        """Add charts for each feature"""
        for result in results:
            sheet_name = result['feature'][0]
            result.to_excel(writer, sheet_name=sheet_name)
            sheet = writer.sheets[sheet_name]

            max_row = len(result)
            col_indexes = {col: result.columns.get_loc(col) + 1 for col in
                         ['event_cnt', 'non_event_cnt', 'score_ball', 'WOE', 'event_rate']}

            events_chart = ExcelBuilder._create_events_chart(workbook, sheet_name,
                                                           max_row, col_indexes)
            score_chart = ExcelBuilder._create_score_chart(workbook, sheet_name,
                                                         max_row, col_indexes)

            events_chart.set_size({'width': width, 'height': height})
            events_chart.set_legend({'position': 'bottom'})
            score_chart.set_size({'width': width, 'height': height})
            score_chart.set_legend({'position': 'bottom'})

            ExcelBuilder._format_feature_sheet(sheet, result, max_row)

            sheet.insert_chart(f'{first_pos}{max_row + 3}', events_chart)
            sheet.insert_chart(f'{second_pos}{max_row + 3}', score_chart)

    @staticmethod
    def _create_events_chart(workbook,
                           sheet_name: str,
                           max_row: int,
                           col_idx: Dict) -> Any:
        """Create events distribution chart"""
        chart = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})

        chart.add_series({
            'name': 'event_cnt',
            'values': [sheet_name, 1, col_idx['event_cnt'], max_row, col_idx['event_cnt']]
        })
        chart.add_series({
            'name': 'non_event_cnt',
            'values': [sheet_name, 1, col_idx['non_event_cnt'], max_row, col_idx['non_event_cnt']]
        })

        woe_line = workbook.add_chart({'type': 'line'})
        woe_line.add_series({
            'name': 'WOE',
            'values': [sheet_name, 1, col_idx['WOE'], max_row, col_idx['WOE']],
            'smooth': False,
            'y2_axis': True
        })

        chart.combine(woe_line)
        return chart

    @staticmethod
    def _create_score_chart(workbook,
                          sheet_name: str,
                          max_row: int,
                          col_idx: Dict) -> Any:
        """Create score distribution chart"""
        chart = workbook.add_chart({'type': 'column'})

        chart.add_series({
            'name': 'score_ball',
            'values': [sheet_name, 1, col_idx['score_ball'], max_row, col_idx['score_ball']]
        })

        rate_line = workbook.add_chart({'type': 'line'})
        rate_line.add_series({
            'name': 'event_rate',
            'values': [sheet_name, 1, col_idx['event_rate'], max_row, col_idx['event_rate']],
            'smooth': False,
            'y2_axis': True
        })

        chart.combine(rate_line)
        return chart

    @staticmethod
    def _format_feature_sheet(sheet,
                            result: pd.DataFrame,
                            max_row: int) -> None:
        """Format individual feature sheet"""
        merge_format = sheet.book.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })

        for col, width in zip([1, 2, 3], [20, 10, 10]):
            sheet.merge_range(1, col, max_row, col, result.iloc[1, col-1], merge_format)
            sheet.set_column(col, col, width)


def save_scorecard(feature_names: List[str],
                  encoder: Any,
                  model_results: pd.DataFrame,
                  base_points: int,
                  odds: int,
                  points_to_double: int,
                  path: str) -> None:
    """Save complete scorecard to Excel"""
    factor = points_to_double / np.log(2)
    offset = base_points - factor * np.log(odds)

    try:
        stats = ScorecardBuilder.build_scorecard(
            feature_names=feature_names,
            encoder=encoder,
            model_results=model_results,
            factor=factor,
            offset=offset
        )

        with pd.ExcelWriter(os.path.join(path, "Scorecard.xlsx"), engine="xlsxwriter") as writer:
            ExcelBuilder.build_excel_sheet(feature_stats=stats, writer=writer)

    except Exception as e:
        print(f"Error saving scorecard: {e}")


def calc_model_results(model: Model) -> pd.DataFrame:
    """Calculate model results summary"""
    results = pd.DataFrame({
        'index': ['const'] + [name[4:] for name in model.feature_names_],
        'coef': [model.intercept_] + list(model.coef_),
        'P>|z|': [0] + list(model.pvalues_)
    })
    return results.reset_index(drop=True)
