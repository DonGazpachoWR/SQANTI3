"""Optimal minimum prevalence: model of QC and "auto" requisites of the filter.

QC fits a mixture of two binomials (noise, epsilon; signal, p) to the
prevalence columns, without the transcripts expressed in no sample. The
rules filter turns "auto" into the smallest number of samples whose false
positive rate is not above alpha (corrected by the number of groups).
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, main_path)

from src.utilities.prevalence_model import (
    FITTED, NOT_IDENTIFIABLE, TOO_FEW_TRANSCRIPTS, auto_columns, binomial_tail,
    fit_binomial_mixture, model_path, optimal_min_prevalence, prevalence_model,
    prevalence_thresholds, read_min_prevalence, resolve_auto, thresholds_path
)
from src.utilities.filter.sqanti3_rules_filter import read_json_rules, rules_filter
from src.qc_output import write_prevalence_model
from src.module_logging import filter_logger


def simulate(rng, n, groups, noise, epsilon, p):
    """Counts (0/1) of n transcripts, each pair (transcript, group) with its class."""
    cols = {}
    for g, m in groups.items():
        is_noise = rng.random(n) < noise
        prob = np.where(is_noise, epsilon, p)
        for j in range(m):
            cols[f"{g}{j + 1}"] = (rng.random(n) < prob).astype(float)
    return pd.DataFrame(cols)


def design_of(groups):
    return {g: [f"{g}{j + 1}" for j in range(m)] for g, m in groups.items()}


# ---------------------------------------------------------------- paths

def test_model_path_next_to_classification():
    assert model_path("out/iso_classification.txt") == "out/iso_prevalence_model.tsv"


def test_thresholds_path_next_to_filter_classification():
    assert (thresholds_path("out/iso_RulesFilter_classification.txt")
            == "out/iso_prevalence_thresholds.tsv")


# ---------------------------------------------------------------- threshold

@pytest.mark.parametrize("samples, epsilon, alpha, expected", [
    (3, 0.05, 0.05, 2), (5, 0.05, 0.05, 2), (3, 0.01, 0.05, 1),
    (5, 0.10, 0.05, 3), (5, 0.20, 0.05, 4), (5, 0.40, 0.05, 5)])
def test_optimal_min_prevalence(samples, epsilon, alpha, expected):
    m = optimal_min_prevalence(samples, epsilon, alpha)
    assert m == expected
    assert binomial_tail(m, samples, epsilon) <= alpha
    if m > 1:
        assert binomial_tail(m - 1, samples, epsilon) > alpha


def test_no_threshold_controls_the_rate():
    assert optimal_min_prevalence(5, 0.6, 0.05) is None


def test_binomial_tail():
    # P(S >= 2) with M = 3, epsilon = 0.05: 3 * 0.05^2 * 0.95 + 0.05^3
    assert binomial_tail(2, 3, 0.05) == pytest.approx(0.00725)


# ---------------------------------------------------------------- fit

def test_shared_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    groups = {"K": 5, "B": 5}
    counts = simulate(rng, 100000, groups, 0.6, 0.1, 0.85)
    expressed = counts.loc[counts.any(axis=1)]
    det = {g: expressed[s].sum(axis=1).to_numpy(dtype=int)
           for g, s in design_of(groups).items()}
    fit = fit_binomial_mixture(det, groups, shared=True)
    for g in groups:
        assert fit[g]["status"] == FITTED
        assert fit[g]["epsilon"] == pytest.approx(0.1, abs=0.005)
        assert fit[g]["p"] == pytest.approx(0.85, abs=0.01)
        assert fit[g]["noise_fraction"] == pytest.approx(0.6, abs=0.01)


def test_fit_is_deterministic():
    rng = np.random.default_rng(3)
    counts = simulate(rng, 5000, {"S": 5}, 0.7, 0.05, 0.8)
    det = {"S": counts.loc[counts.any(axis=1)].sum(axis=1).to_numpy(dtype=int)}
    assert (fit_binomial_mixture(det, {"S": 5}, True)
            == fit_binomial_mixture(det, {"S": 5}, True))


def test_three_samples_alone_are_not_identifiable():
    # Without the all-zero cell, 3 samples leave 2 free cells for 3 parameters
    det = {"S": np.repeat([1, 2, 3], 100)}
    fit = fit_binomial_mixture(det, {"S": 3}, shared=True)
    assert fit["S"]["status"] == NOT_IDENTIFIABLE
    assert np.isnan(fit["S"]["epsilon"])


def test_too_few_transcripts():
    fit = fit_binomial_mixture({"S": np.array([1, 2, 5])}, {"S": 5}, True)
    assert fit["S"]["status"] == TOO_FEW_TRANSCRIPTS


# ---------------------------------------------------------------- model

def test_model_rows_and_excluded_transcripts():
    rng = np.random.default_rng(2)
    groups = {"K": 3, "B": 3}
    counts = simulate(rng, 50000, groups, 0.7, 0.03, 0.9)
    model = prevalence_model(counts, design_of(groups), 0)
    assert model["column"].tolist() == ["prevalence", "prevalence_K", "prevalence_B"]
    excluded = int((~counts.astype(bool).any(axis=1)).sum())
    assert (model["excluded"] == excluded).all()
    assert (model["transcripts"] == len(counts) - excluded).all()
    groups_rows = model.iloc[1:]
    # Common estimate for the groups, and one estimate per group
    assert groups_rows["epsilon"].nunique() == 1
    assert groups_rows["epsilon"].iloc[0] == pytest.approx(0.03, abs=0.003)
    assert (groups_rows["status_group"] == FITTED).all()
    assert groups_rows["epsilon_group"].nunique() == 2


def test_truncation_on_the_samples_of_each_column():
    # A transcript seen only in a sample outside the groups (e.g. a mixture)
    # counts for prevalence but is left out of the model of the groups
    counts = pd.DataFrame({"K1": [1, 0, 0], "K2": [1, 0, 0], "MIX": [1, 1, 0]})
    model = prevalence_model(counts, {"K": ["K1", "K2"]}, 0).set_index("column")
    assert model.loc["prevalence", ["transcripts", "excluded"]].tolist() == [2, 1]
    assert model.loc["prevalence_K", ["transcripts", "excluded"]].tolist() == [1, 2]


def test_model_uses_min_expression():
    counts = pd.DataFrame({"S1": [0.5, 2], "S2": [0.5, 2], "S3": [0.5, 2],
                           "S4": [0.5, 2]})
    assert prevalence_model(counts, {}, 0)["excluded"].iloc[0] == 0
    assert prevalence_model(counts, {}, 1)["excluded"].iloc[0] == 1


# ---------------------------------------------------------------- thresholds

def model_table(epsilon=0.1, p=0.9, status=FITTED):
    return pd.DataFrame({
        "column": ["prevalence", "prevalence_K", "prevalence_B"],
        "samples": [10, 5, 5], "epsilon": [epsilon] * 3, "p": [p] * 3,
        "status": [status] * 3})


def test_thresholds_correct_alpha_by_the_groups():
    t = prevalence_thresholds(model_table(), ["prevalence", "prevalence_K"], 0.05)
    t = t.set_index("column")
    assert t.loc["prevalence", "alpha_column"] == pytest.approx(0.05)
    assert t.loc["prevalence_K", "alpha_column"] == pytest.approx(1 - 0.95 ** 0.5)
    assert t.loc["prevalence_K", "min_prevalence"] == 3
    assert t.loc["prevalence_K", "fpr"] == pytest.approx(binomial_tail(3, 5, 0.1))
    assert t.loc["prevalence_K", "power"] == pytest.approx(binomial_tail(3, 5, 0.9))
    assert t["controlled"].all()


def test_uncontrolled_rate_requires_all_samples():
    t = prevalence_thresholds(model_table(epsilon=0.7), ["prevalence_K"], 0.01)
    assert t["min_prevalence"].iloc[0] == 5
    assert not t["controlled"].iloc[0]


def test_threshold_of_a_column_missing_from_the_model():
    with pytest.raises(ValueError):
        prevalence_thresholds(model_table(), ["prevalence_X"], 0.01)


def test_threshold_of_an_unfitted_model():
    with pytest.raises(ValueError):
        prevalence_thresholds(model_table(status=NOT_IDENTIFIABLE), ["prevalence"], 0.01)


# ---------------------------------------------------------------- "auto"

def test_auto_columns():
    rules = {"genic": [{"prevalence_K": "auto", "min_cov": 3}, {"prevalence_B": "AUTO"}],
             "rest": [{"prevalence": 2}]}
    assert auto_columns(rules) == ["prevalence_B", "prevalence_K"]


def test_resolve_auto():
    rules = {"genic": [{"prevalence_K": "auto", "min_cov": 3}]}
    assert resolve_auto(rules, {"prevalence_K": 4}) == {"genic": [{"prevalence_K": 4, "min_cov": 3}]}
    assert rules["genic"][0]["prevalence_K"] == "auto"


def test_auto_only_for_prevalence():
    with pytest.raises(ValueError):
        resolve_auto({"genic": [{"FL": "auto"}]}, {"FL": 2})


def test_auto_without_threshold():
    with pytest.raises(ValueError):
        resolve_auto({"genic": [{"prevalence_K": "auto"}]}, None)


def test_read_json_rules_resolves_auto(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({"genic": [{"prevalence_K": "auto"}]}))
    rules = read_json_rules(str(p), {"prevalence_K": 3})["genic"][0]
    assert rules.to_dict("records") == [{"column": "prevalence_K", "type": "Min_Threshold", "rule": 3}]


def test_read_json_rules_auto_without_threshold_exits(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({"genic": [{"prevalence_K": "auto"}]}))
    with pytest.raises(SystemExit):
        read_json_rules(str(p))


# ---------------------------------------------------------------- filter

def write_classification(path):
    pd.DataFrame({
        "isoform": ["t1", "t2", "t3"], "structural_category": ["genic"] * 3,
        "exons": [3, 3, 3], "prevalence_K": [5, 2, 3], "prevalence_B": [0, 1, 1],
    }).to_csv(path, sep="\t", index=False)


def test_rules_filter_with_auto(tmp_path):
    class_file = tmp_path / "iso_classification.txt"
    write_classification(class_file)
    model_table(epsilon=0.1).to_csv(model_path(str(class_file)), sep="\t", index=False)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"genic": [{"prevalence_K": "auto"}, {"prevalence_B": "auto"}]}))
    prefix = str(tmp_path / "iso")
    rules_filter(str(class_file), str(rules), False, prefix, filter_logger,
                 model_path(str(class_file)), 0.05)

    out = pd.read_csv(f"{prefix}_RulesFilter_classification.txt", sep="\t")
    # m = 3 for each group (epsilon 0.1, 5 samples, alpha corrected for 2 groups)
    assert out["filter_result"].tolist() == ["Isoform", "Artifact", "Isoform"]
    thresholds = pd.read_csv(f"{prefix}_prevalence_thresholds.tsv", sep="\t")
    assert thresholds["min_prevalence"].tolist() == [3, 3]
    assert thresholds["isoforms_passing"].tolist() == [0, 2]
    assert read_min_prevalence(f"{prefix}_prevalence_thresholds.tsv") == {
        "prevalence_B": 3, "prevalence_K": 3}


def test_rules_filter_without_auto_writes_no_thresholds(tmp_path):
    class_file = tmp_path / "iso_classification.txt"
    write_classification(class_file)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"genic": [{"prevalence_K": 3}]}))
    prefix = str(tmp_path / "iso")
    rules_filter(str(class_file), str(rules), False, prefix, filter_logger)
    assert not os.path.exists(f"{prefix}_prevalence_thresholds.tsv")


def test_rules_filter_auto_without_model_exits(tmp_path):
    class_file = tmp_path / "iso_classification.txt"
    write_classification(class_file)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"genic": [{"prevalence_K": "auto"}]}))
    with pytest.raises(SystemExit):
        rules_filter(str(class_file), str(rules), False, str(tmp_path / "iso"), filter_logger,
                     model_path(str(class_file)), 0.05)


# ---------------------------------------------------------------- QC

def qc_isoforms(counts, design):
    return {iso: SimpleNamespace(FL_dict=row.to_dict(), counts_design=design, min_expression=0)
            for iso, row in counts.iterrows()}


def test_qc_writes_the_model(tmp_path):
    rng = np.random.default_rng(4)
    groups = {"K": 4, "B": 4}
    counts = simulate(rng, 2000, groups, 0.6, 0.05, 0.9)
    class_file = str(tmp_path / "iso_classification.txt")
    write_prevalence_model(qc_isoforms(counts, design_of(groups)), class_file)
    model = pd.read_csv(model_path(class_file), sep="\t")
    assert model["column"].tolist() == ["prevalence", "prevalence_K", "prevalence_B"]
    assert (model.loc[1:, "status"] == FITTED).all()


def test_qc_single_sample_writes_no_model(tmp_path):
    isoforms = {"t1": SimpleNamespace(FL_dict={}, counts_design={}, min_expression=0)}
    class_file = str(tmp_path / "iso_classification.txt")
    write_prevalence_model(isoforms, class_file)
    assert not os.path.exists(model_path(class_file))
