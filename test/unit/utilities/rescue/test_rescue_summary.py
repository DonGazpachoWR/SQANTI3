"""Tests for the rescue summary tables (numbers shown in the rescue report)."""
import os
import sys

import pandas as pd
import pytest

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, main_path)

from src.utilities.rescue.rescue_summary import (
    artifact_outcomes, gene_recovery, rules_failed_requisites,
    summarize_rescue, write_rescue_summary, hit_targets_table
)


@pytest.fixture
def classif():
    rows = [
        # isoform, category, exons, filter_result, associated_transcript, associated_gene
        ("fsm_ok", "full-splice_match", 3, "Isoform", "R1", "G1"),
        ("fsm_art_repr", "full-splice_match", 3, "Artifact", "R1", "G1"),   # R1 represented
        ("fsm_art_lost", "full-splice_match", 4, "Artifact", "R2", "G2"),   # automatic rescue
        ("fsm_art_lost2", "full-splice_match", 4, "Artifact", "R2", "G2"),  # same reference
        ("fsm_mono", "full-splice_match", 1, "Artifact", "R6", "G6"),
        ("nic_ref", "novel_in_catalog", 5, "Artifact", "novel", "G3"),       # rescued by reference
        ("nnc_lr", "novel_not_in_catalog", 5, "Artifact", "novel", "G1"),    # rescued by long read
        ("nic_nohit", "novel_in_catalog", 2, "Artifact", "novel", "G4"),
        ("nnc_fail", "novel_not_in_catalog", 2, "Artifact", "novel", "G5"),
        ("ism_repr", "incomplete-splice_match", 2, "Artifact", "R1", "G1"),
        ("genic_art", "genic", 2, "Artifact", "novel", "G7"),
        ("lr_target", "novel_not_in_catalog", 6, "Isoform", "novel", "G1"),
        ("novel_gene", "intergenic", 2, "Artifact", "novel", "novelGene_1"),
    ]
    return pd.DataFrame(rows, columns=["isoform", "structural_category", "exons", "filter_result",
                                       "associated_transcript", "associated_gene"])


@pytest.fixture
def rescue_df():
    return pd.DataFrame([
        ("fsm_art_lost", "R2", "automatic", "reference", "yes"),
        ("fsm_art_lost2", "R2", "automatic", "reference", "no"),
        ("nic_ref", "R3", "rules_mapping", "reference", "yes"),
        ("nnc_lr", "lr_target", "rules_mapping", "lr_defined", "no"),
    ], columns=["artifact", "assigned_transcript", "rescue_mode", "origin", "reintroduced"])


@pytest.fixture
def hits():
    return pd.DataFrame([
        ("nic_ref", "R3", "reference", "Isoform", pd.NA),
        ("nic_ref", "R4", "reference", "Artifact", "min_cov: 0 < 3; NA value in FL"),
        ("nnc_lr", "lr_target", "lr_defined", "Isoform", pd.NA),
        ("nnc_fail", "R4", "reference", "Artifact", "min_cov: 0 < 3; NA value in FL"),
        ("nnc_fail", "R5", "reference", "Artifact", "min_cov: 1 < 3"),
        ("nic_nohit", None, None, None, pd.NA),  # unmapped candidate
    ], columns=["rescue_candidate", "mapping_hit", "hit_origin", "hit_filter_result",
                "hit_filter_reason"])


CANDIDATES = ["nic_ref", "nnc_lr", "nic_nohit", "nnc_fail"]


def get(summary, section, group, category, field="count"):
    row = summary[(summary.section == section) & (summary.group == group) &
                  (summary.category == category)]
    assert len(row) == 1, f"{section}/{group}/{category}"
    return row.iloc[0][field]


class TestArtifactOutcomes:
    def test_full_mode(self, classif, rescue_df, hits):
        out = artifact_outcomes(classif, rescue_df, "full", "all", CANDIDATES, hits)
        got = dict(zip(out.artifact, out.outcome))
        assert got == {
            "fsm_art_repr": "reference_already_represented",
            "fsm_art_lost": "rescued_reference",
            "fsm_art_lost2": "rescued_reference",
            "fsm_mono": "reference_already_represented",
            "nic_ref": "rescued_reference",
            "nnc_lr": "rescued_lr_defined",
            "nic_nohit": "no_mapping_hit",
            "nnc_fail": "no_hit_passes_filter",
            "ism_repr": "reference_already_represented",
            "genic_art": "category_not_eligible",
            "novel_gene": "category_not_eligible",
        }

    def test_automatic_mode(self, classif, rescue_df):
        auto = rescue_df[rescue_df.rescue_mode == "automatic"]
        out = artifact_outcomes(classif, auto, "automatic", "none")
        got = dict(zip(out.artifact, out.outcome))
        assert got["nic_ref"] == "mapping_not_run"
        assert got["ism_repr"] == "mapping_not_run"
        assert got["fsm_mono"] == "mono_exonic_excluded"
        assert got["genic_art"] == "category_not_eligible"

    def test_mono_exonic_candidates_excluded(self, classif, rescue_df):
        classif = classif.copy()
        classif.loc[classif.isoform == "nic_nohit", "exons"] = 1
        out = artifact_outcomes(classif, rescue_df, "full", "fsm",
                                [c for c in CANDIDATES if c != "nic_nohit"], None)
        assert out.set_index("artifact").loc["nic_nohit", "outcome"] == "mono_exonic_excluded"


class TestSummary:
    @pytest.fixture
    def summary(self, classif, rescue_df, hits):
        ref_genes = {"R1": "G1", "R2": "G2", "R3": "G3", "R4": "G5", "R5": "G5", "R6": "G6"}
        summary, _ = summarize_rescue(classif, rescue_df, pd.Series(["R2", "R3"]), "full",
                                      "rules", "all", CANDIDATES, hits, ref_genes)
        return summary

    def test_overview_units(self, summary):
        assert get(summary, "overview", "artifacts", "artifacts") == 11
        # rescued (4) + no hit (1) + failing hits (1)
        assert get(summary, "overview", "artifacts", "considered_for_rescue") == 6
        assert get(summary, "overview", "artifacts", "rescued") == 4
        assert get(summary, "overview", "transcripts", "assigned_transcripts") == 3
        assert get(summary, "overview", "transcripts", "added_reference_transcripts") == 2

    def test_mode_origin(self, summary):
        assert get(summary, "mode_origin_artifacts", "automatic", "reference") == 2
        assert get(summary, "mode_origin_artifacts", "rules_mapping", "reference") == 1
        assert get(summary, "mode_origin_artifacts", "rules_mapping", "lr_defined") == 1
        assert get(summary, "mode_origin_artifacts", "automatic", "reference", "percent") == 50.0
        assert get(summary, "mode_origin_added", "automatic", "reference") == 1
        assert get(summary, "mode_origin_added", "rules_mapping", "reference") == 1

    def test_outcome_percent_is_over_the_category(self, summary):
        assert get(summary, "artifact_outcome", "full-splice_match", "rescued_reference") == 2
        assert get(summary, "artifact_outcome", "full-splice_match", "rescued_reference", "total") == 4
        assert get(summary, "artifact_outcome", "full-splice_match", "rescued_reference", "percent") == 50.0

    def test_composition_labels_added_by_artifact_category(self, summary):
        assert get(summary, "composition", "before", "full-splice_match") == 1
        assert get(summary, "composition", "after", "rescued_from:full-splice_match") == 1
        assert get(summary, "composition", "after", "rescued_from:novel_in_catalog") == 1
        assert get(summary, "composition", "after", "full-splice_match", "total") == 4

    def test_gene_recovery(self, summary):
        # G1 has isoforms; G2..G7 are lost; novel genes are left out
        assert get(summary, "gene_recovery", "genes", "genes_lost_by_filter") == 6
        assert get(summary, "gene_recovery", "lost_genes", "recovered_by_rescue") == 2

    def test_multiplicity(self, summary):
        # R2 receives two artifacts, R3 and lr_target one each
        assert get(summary, "multiplicity", "reference", "2") == 1
        assert get(summary, "multiplicity", "reference", "1") == 1
        assert get(summary, "multiplicity", "lr_defined", "1") == 1

    def test_exon_class(self, summary):
        assert get(summary, "exon_class", "mono-exonic", "not_rescued") == 1
        assert get(summary, "exon_class", "multi-exonic", "rescued") == 4

    def test_failed_requisites(self, summary):
        # R4 and R5 fail; R4 counted once although two candidates hit it
        assert get(summary, "failed_requisites", "reference_targets", "min_cov") == 2
        assert get(summary, "failed_requisites", "reference_targets", "FL (NA)") == 1
        assert get(summary, "target_filter", "reference", "Artifact") == 2

    def test_ml_threshold(self, classif, rescue_df, hits):
        ml_hits = hits.drop(columns="hit_filter_reason").assign(
            hit_POS_MLprob=[0.9, 0.2, 0.8, 0.2, 0.5, None])
        summary, _ = summarize_rescue(classif, rescue_df, ["R2", "R3"], "full", "ml", "all",
                                      CANDIDATES, ml_hits, thr=0.7)
        assert get(summary, "ml_threshold", "reference_targets", "above_threshold") == 1
        assert get(summary, "ml_threshold", "reference_targets", "below_threshold") == 2


def test_gene_recovery_without_reference_genes(classif, rescue_df):
    genes = gene_recovery(classif, rescue_df, ["R2", "R3"])
    assert genes["recovered_genes"] == {"G2", "G3"}


def test_failed_requisites_counts_each_target_once(hits):
    counts, n_failed = rules_failed_requisites(hits)
    assert n_failed == 2
    assert counts == {"min_cov": 2, "FL (NA)": 1}


def test_empty_automatic_rescue(classif):
    # automatic rescue returns an empty DataFrame when no reference is lost
    empty = pd.DataFrame(columns=["artifact", "assigned_transcript", "rescue_mode",
                                  "origin", "reintroduced"])
    summary, outcomes = summarize_rescue(classif, empty, pd.DataFrame(columns=["isoform"]),
                                         "automatic")
    assert get(summary, "overview", "artifacts", "rescued") == 0
    assert get(summary, "overview", "transcripts", "added_reference_transcripts") == 0
    assert (outcomes.outcome != "rescued_reference").all()


def test_hit_targets_table_rules(tmp_path, classif):
    ref = tmp_path / "ref_class.tsv"
    pd.DataFrame({"isoform": ["R3", "R4"], "filter_result": ["Isoform", "Artifact"]}).to_csv(
        ref, sep="\t", index=False)
    reasons = tmp_path / "reasons.tsv"
    pd.DataFrame({"isoform": ["R4"], "structural_category": ["full-splice_match"],
                  "filter_reason": ["min_cov: 0 < 3"]}).to_csv(reasons, sep="\t", index=False)
    hits_df = pd.DataFrame({"rescue_candidate": ["nic_ref", "nic_ref", "nnc_lr"],
                            "mapping_hit": ["R3", "R4", "lr_target"],
                            "alignment_type": [0, 0, 0], "alignment_score": [10, 9, 8]})
    out = hit_targets_table(hits_df, classif, str(ref), "rules", reasons_file=str(reasons))
    got = out.set_index("mapping_hit")
    assert got.loc["R3", "hit_filter_result"] == "Isoform"
    assert got.loc["R4", "hit_filter_reason"] == "min_cov: 0 < 3"
    assert got.loc["lr_target", "hit_origin"] == "lr_defined"


def test_write_rescue_summary(tmp_path, classif, rescue_df, hits):
    summary, outcomes = summarize_rescue(classif, rescue_df, ["R2", "R3"], "full", "rules",
                                         "all", CANDIDATES, hits)
    files = write_rescue_summary(str(tmp_path / "x"), summary, outcomes, hits, "full", "rules")
    written = pd.read_csv(files["summary"], sep="\t")
    assert list(written.columns) == ["section", "group", "category", "count", "total", "percent"]
    assert written[written.section == "run"].category.tolist() == ["full", "rules"]
    targets = pd.read_csv(files["hits"], sep="\t")
    assert targets.mapping_hit.is_unique and targets.mapping_hit.notna().all()
    assert len(pd.read_csv(files["outcomes"], sep="\t")) == 11


def test_hit_targets_table_without_reference_artifacts(tmp_path, classif):
    # When every reference transcript passes, the reasons file has no filter_reason column
    ref = tmp_path / "ref_class.tsv"
    pd.DataFrame({"isoform": ["R3"], "filter_result": ["Isoform"]}).to_csv(ref, sep="\t", index=False)
    reasons = tmp_path / "reasons.tsv"
    pd.DataFrame(columns=["isoform", "chrom", "structural_category"]).to_csv(reasons, sep="\t", index=False)
    hits_df = pd.DataFrame({"rescue_candidate": ["nic_ref"], "mapping_hit": ["R3"],
                            "alignment_type": [0], "alignment_score": [10]})
    out = hit_targets_table(hits_df, classif, str(ref), "rules", reasons_file=str(reasons))
    assert out["hit_filter_reason"].isna().all()
