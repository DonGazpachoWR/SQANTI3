"""
prevalence_model.py

Optimal minimum prevalence for the rules filter. The number of samples
where a transcript is expressed is modelled as a mixture of two binomials:
technical noise, detected in each sample with probability epsilon, and
biological signal, detected with probability p. QC estimates epsilon, p
and the fraction of noise transcripts by EM on the prevalence columns,
leaving out the transcripts not expressed in any sample (zero-truncated
likelihood). With the per-group columns of a counts design, epsilon, p
and the noise fraction are common to all groups, each group with its own
number of samples; the estimates of each group are reported too. The
filter turns a rule value "auto" into the smallest number of samples m
whose false positive rate, P(S >= m | noise), is not above alpha,
corrected by the number of groups since rules combine them with OR.

Version: 1.0
Author: Adrian Berenguer-Agustí (I2SysBio-CSIC)
Date: 2026-10-08
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.utils import is_expressed

# Value of a prevalence requisite in the rules JSON that asks for the
# optimal minimum number of samples.
AUTO_VALUE = "auto"
# Prevalence over all samples; per-group columns are prevalence_<group>.
PREVALENCE_COLUMN = "prevalence"
GROUP_PREFIX = f"{PREVALENCE_COLUMN}_"
# Suffixes of the model (QC) and thresholds (filter) files.
MODEL_SUFFIX = "_prevalence_model.tsv"
THRESHOLDS_SUFFIX = "_prevalence_thresholds.tsv"
# Starting point of the EM (noise fraction, epsilon, p), fixed so that the
# fit is deterministic.
EM_START = (0.5, 0.05, 0.7)
# Maximum EM iterations and change in log-likelihood taken as convergence.
EM_MAX_ITER = 10000
EM_TOLERANCE = 1e-10
# Probabilities are kept inside (EPS_BOUND, 1 - EPS_BOUND) during the EM.
EPS_BOUND = 1e-9
# Minimum number of transcripts expressed in some sample to fit the model.
MIN_MODEL_TRANSCRIPTS = 100
# Status of a fit in the model file.
FITTED = "fitted"
NOT_CONVERGED = "not_converged"
NOT_IDENTIFIABLE = "not_identifiable"
TOO_FEW_TRANSCRIPTS = "too_few_transcripts"


def model_path(classification_path: str) -> str:
    """Path of the prevalence model written next to a QC classification.

    Args:
        classification_path: QC classification file
            (<prefix>_classification.txt).

    Returns:
        Path <prefix>_prevalence_model.tsv in the same directory.
    """
    base = classification_path
    if base.endswith("_classification.txt"):
        base = base[:-len("_classification.txt")]
    else:
        base = os.path.splitext(base)[0]
    result = f"{base}{MODEL_SUFFIX}"
    return result


def thresholds_path(filter_classification_path: str) -> str:
    """Path of the prevalence thresholds written by the rules filter.

    Args:
        filter_classification_path: rules filter classification file
            (<prefix>_RulesFilter_classification.txt).

    Returns:
        Path <prefix>_prevalence_thresholds.tsv in the same directory.
    """
    base = filter_classification_path
    if base.endswith("_RulesFilter_classification.txt"):
        base = base[:-len("_RulesFilter_classification.txt")]
    else:
        base = os.path.splitext(base)[0]
    result = f"{base}{THRESHOLDS_SUFFIX}"
    return result


def is_prevalence_column(column: str) -> bool:
    """Whether a column is prevalence or prevalence_<group>."""
    result = column == PREVALENCE_COLUMN or column.startswith(GROUP_PREFIX)
    return result


def is_auto(value) -> bool:
    """Whether a rule value asks for the optimal minimum prevalence."""
    result = isinstance(value, str) and value.strip().lower() == AUTO_VALUE
    return result


def auto_columns(json_data: dict) -> list[str]:
    """Columns set to "auto" in any rule of a rules JSON.

    Args:
        json_data: rules JSON as loaded, structural category -> list of
            rules (dicts column -> value).

    Returns:
        Sorted names of the columns with value "auto".
    """
    columns = {column for rules in json_data.values() for rule in rules
               for column, value in rule.items() if is_auto(value)}
    result = sorted(columns)
    return result


def resolve_auto(json_data: dict,
                 min_prevalence: dict[str, int] | None) -> dict:
    """Replace the "auto" values of a rules JSON by their thresholds.

    Args:
        json_data: rules JSON as loaded, structural category -> list of
            rules.
        min_prevalence: column -> minimum number of samples, as computed by
            prevalence_thresholds(). Not needed if no rule uses "auto".

    Returns:
        A copy of json_data with every "auto" replaced by its integer.

    Raises:
        ValueError: if "auto" is used in a column other than prevalence or
            prevalence_<group>, or no threshold was computed for it.
    """
    result = {}
    for category, rules in json_data.items():
        result[category] = []
        for rule in rules:
            resolved = dict(rule)
            for column, value in rule.items():
                if not is_auto(value):
                    continue
                if not is_prevalence_column(column):
                    raise ValueError(
                        f"'{AUTO_VALUE}' is only valid for prevalence and "
                        f"prevalence_<group>, not for '{column}'.")
                if min_prevalence is None or column not in min_prevalence:
                    raise ValueError(
                        f"No optimal minimum prevalence was computed for "
                        f"'{column}'.")
                resolved[column] = int(min_prevalence[column])
            result[category].append(resolved)
    return result


def binomial_tail(m: int, samples: int, prob: float) -> float:
    """P(S >= m) for S ~ Binomial(samples, prob)."""
    result = float(binom.sf(m - 1, samples, prob))
    return result


def optimal_min_prevalence(samples: int, epsilon: float,
                           alpha: float) -> int | None:
    """Smallest number of samples whose false positive rate is <= alpha.

    m* = min{m in 1..M : P(S >= m | noise) <= alpha}. It also gives the
    highest power among the thresholds that control the error, since both
    tails decrease with m.

    Args:
        samples: number of samples M of the column.
        epsilon: probability that a noise transcript is expressed in a
            sample.
        alpha: tolerated false positive rate for the column.

    Returns:
        m*, or None if not even m = M keeps the rate below alpha.
    """
    result = None
    for m in range(1, samples + 1):
        if binomial_tail(m, samples, epsilon) <= alpha:
            result = m
            break
    return result


def _mixture_pmf(samples: int, noise: float, epsilon: float,
                 p: float) -> tuple[np.ndarray, np.ndarray]:
    """Noise and signal terms of the mixture for S = 0..samples."""
    k = np.arange(samples + 1)
    noise_term = noise * binom.pmf(k, samples, epsilon)
    signal_term = (1 - noise) * binom.pmf(k, samples, p)
    result = (noise_term, signal_term)
    return result


def _is_identifiable(samples: list[int], n_params: int) -> bool:
    """Whether the zero-truncated joint table has enough free cells.

    The table of detections per group has prod(M_g + 1) cells; the
    all-zero cell is not observed and the total is fixed.
    """
    cells = math.prod(m + 1 for m in samples)
    result = cells - 2 >= n_params
    return result


def fit_binomial_mixture(detections: dict[str, np.ndarray],
                         samples: dict[str, int],
                         shared: bool) -> dict[str, dict]:
    """Fit the noise/signal binomial mixture to the detections per group.

    Only transcripts expressed in some sample are given, so the likelihood
    is zero-truncated: the transcripts not expressed in any group are
    treated as missing data, whose expected number is added to the zero
    class of every group at each iteration (Dempster, Laird and Rubin,
    1977). Each pair (transcript, group) has its own latent class, since a
    transcript can be expressed in one group only.

    Args:
        detections: group -> number of samples of the group where each
            transcript is expressed (same transcripts, same order).
        samples: group -> number of samples of the group.
        shared: whether noise fraction, epsilon and p are common to all
            groups (True) or estimated for each group (False).

    Returns:
        group -> {"noise_fraction", "epsilon", "p", "status"}, with NaN
        estimates when the status is not FITTED.
    """
    groups = list(detections)
    n_transcripts = len(next(iter(detections.values()))) if groups else 0
    n_params = 3 if shared else 3 * len(groups)
    histograms = {g: np.bincount(detections[g], minlength=samples[g] + 1)
                  .astype(float) for g in groups}
    params = {g: list(EM_START) for g in groups}

    status = FITTED
    if n_transcripts < MIN_MODEL_TRANSCRIPTS:
        status = TOO_FEW_TRANSCRIPTS
    elif not _is_identifiable([samples[g] for g in groups], n_params):
        status = NOT_IDENTIFIABLE
    else:
        status = NOT_CONVERGED
        previous = -np.inf
        for _ in range(EM_MAX_ITER):
            terms = {g: _mixture_pmf(samples[g], *params[g]) for g in groups}
            pmf = {g: terms[g][0] + terms[g][1] for g in groups}
            p_zero = math.prod(float(pmf[g][0]) for g in groups)
            loglik = (sum(float(histograms[g] @ np.log(pmf[g]))
                          for g in groups)
                      - n_transcripts * math.log1p(-p_zero))
            if abs(loglik - previous) < EM_TOLERANCE * max(1, abs(loglik)):
                status = FITTED
                break
            previous = loglik

            # E-step: expected unobserved transcripts go to S = 0.
            missing = n_transcripts * p_zero / (1 - p_zero)
            stats = {}
            for g in groups:
                complete = histograms[g].copy()
                complete[0] += missing
                resp = terms[g][0] / pmf[g]
                k = np.arange(samples[g] + 1)
                stats[g] = np.array([
                    complete @ resp, complete @ (1 - resp),
                    complete @ (resp * k), complete @ ((1 - resp) * k),
                    (complete @ resp) * samples[g],
                    (complete @ (1 - resp)) * samples[g]])

            # M-step, pooling the sufficient statistics if shared.
            pools = [groups] if shared else [[g] for g in groups]
            for pool in pools:
                s = sum(stats[g] for g in pool)
                noise = s[0] / (s[0] + s[1])
                epsilon = s[2] / s[4]
                p = s[3] / s[5]
                if epsilon > p:
                    noise, epsilon, p = 1 - noise, p, epsilon
                new = [float(np.clip(x, EPS_BOUND, 1 - EPS_BOUND))
                       for x in (noise, epsilon, p)]
                for g in pool:
                    params[g] = new

    result = {}
    for g in groups:
        values = params[g] if status == FITTED else [np.nan] * 3
        result[g] = {"noise_fraction": values[0], "epsilon": values[1],
                     "p": values[2], "status": status}
    return result


def _detections(expressed: pd.DataFrame,
                columns: dict[str, list[str]]) -> tuple[dict, dict, dict]:
    """Detections per column of the transcripts expressed in its samples.

    The model is zero-truncated on the samples it describes, so only the
    transcripts expressed in at least one of them are kept.

    Returns:
        (detections, samples, counts), with counts holding the number of
        transcripts kept ("transcripts") and left out ("excluded").
    """
    all_samples = [s for samples in columns.values() for s in samples]
    kept = expressed[all_samples].any(axis=1)
    detections = {c: expressed.loc[kept, s].sum(axis=1).to_numpy(dtype=int)
                  for c, s in columns.items()}
    samples = {c: len(s) for c, s in columns.items()}
    counts = {"transcripts": int(kept.sum()), "excluded": int((~kept).sum())}
    result = (detections, samples, counts)
    return result


def prevalence_model(counts: pd.DataFrame, design: dict[str, list[str]],
                     min_expression: float) -> pd.DataFrame:
    """Estimate the prevalence model of a multi-sample count matrix.

    Args:
        counts: isoforms x samples counts (missing counts as 0).
        design: group -> samples (--counts_design); empty without design.
        min_expression: --min_expression of QC (see is_expressed()).

    Returns:
        One row per prevalence column (prevalence, then prevalence_<group>)
        with: column, samples, min_expression, transcripts (expressed in
        some sample of the column, or of any group for prevalence_<group>),
        excluded (expressed in none), noise_fraction, epsilon, p and status
        of the estimate used for the thresholds (common to all groups for
        prevalence_<group>), and the estimates of each group alone
        (noise_fraction_group, epsilon_group, p_group, status_group; NA for
        prevalence).
    """
    expressed = is_expressed(counts, min_expression)

    det, n_samples, kept = _detections(
        expressed, {PREVALENCE_COLUMN: list(counts.columns)})
    fit_all = fit_binomial_mixture(det, n_samples, shared=True)
    rows = [{"column": PREVALENCE_COLUMN,
             "samples": n_samples[PREVALENCE_COLUMN],
             "min_expression": min_expression, **kept,
             **fit_all[PREVALENCE_COLUMN], "noise_fraction_group": np.nan,
             "epsilon_group": np.nan, "p_group": np.nan,
             "status_group": "NA"}]

    if design:
        det, n_samples, kept = _detections(
            expressed, {g: list(s) for g, s in design.items()})
        joint = fit_binomial_mixture(det, n_samples, shared=True)
        alone = fit_binomial_mixture(det, n_samples, shared=False)
        for g in design:
            rows.append({
                "column": f"{GROUP_PREFIX}{g}", "samples": n_samples[g],
                "min_expression": min_expression, **kept, **joint[g],
                "noise_fraction_group": alone[g]["noise_fraction"],
                "epsilon_group": alone[g]["epsilon"],
                "p_group": alone[g]["p"],
                "status_group": alone[g]["status"]})

    result = pd.DataFrame(rows)
    return result


def prevalence_thresholds(model: pd.DataFrame, columns: list[str],
                          alpha: float) -> pd.DataFrame:
    """Optimal minimum prevalence of the requested columns.

    The rate of each prevalence_<group> column is corrected by the number
    of groups G of the model, alpha_column = 1 - (1 - alpha)^(1/G), since
    rules combine the groups with OR. The prevalence column uses alpha.
    When no threshold controls the rate, all the samples are required
    (controlled is False).

    Args:
        model: prevalence model written by QC (prevalence_model()).
        columns: prevalence columns set to "auto" in the rules.
        alpha: tolerated false positive rate per transcript.

    Returns:
        One row per column: column, samples, epsilon, p, alpha,
        alpha_column, min_prevalence, fpr, power and controlled.

    Raises:
        ValueError: if a column is not in the model or its estimate could
            not be fitted.
    """
    indexed = model.set_index("column")
    n_groups = int(model["column"].str.startswith(GROUP_PREFIX).sum())
    rows = []
    for column in columns:
        if column not in indexed.index:
            raise ValueError(
                f"'{column}' is not in the prevalence model of QC: run QC "
                "with a multi-sample --fl_count (and --counts_design for "
                "prevalence_<group>).")
        entry = indexed.loc[column]
        if entry["status"] != FITTED:
            raise ValueError(
                f"The prevalence model of '{column}' could not be fitted "
                f"({entry['status']}): set its minimum prevalence in the "
                "rules instead of 'auto'.")
        n_samples = int(entry["samples"])
        n_tests = n_groups if column.startswith(GROUP_PREFIX) else 1
        alpha_column = 1 - (1 - alpha) ** (1 / n_tests)
        m = optimal_min_prevalence(n_samples, entry["epsilon"], alpha_column)
        controlled = m is not None
        if not controlled:
            m = n_samples
        rows.append({
            "column": column, "samples": n_samples,
            "epsilon": entry["epsilon"], "p": entry["p"], "alpha": alpha,
            "alpha_column": alpha_column, "min_prevalence": m,
            "fpr": binomial_tail(m, n_samples, entry["epsilon"]),
            "power": binomial_tail(m, n_samples, entry["p"]),
            "controlled": controlled})
    result = pd.DataFrame(rows)
    return result


def read_min_prevalence(path: str) -> dict[str, int]:
    """Column -> min_prevalence from a thresholds file of the filter."""
    table = pd.read_csv(path, sep="\t")
    result = {c: int(m) for c, m in zip(table["column"],
                                        table["min_prevalence"])}
    return result
