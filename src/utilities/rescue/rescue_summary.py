"""
Summary of a SQANTI3 rescue run.

Computes the numbers shown in the rescue report and writes them as
machine-readable tables, so that the counts can be tested without R and the
R report (SQANTI3_rescue_report.R) only has to draw them.

Three units are kept apart, since several artifacts can point to the same target:
  - artifacts: isoforms classified as Artifact by the filter
  - assigned transcripts: targets (reference or long-read defined) that rescued artifacts point to
  - added transcripts: reference transcripts reintroduced in the transcriptome (inclusion list)

Each percentage in the summary table is computed over the denominator given in its
'total' column.
"""
import pandas as pd

from src.utilities.rescue.rescue_by_mapping import merge_classifications, add_filter_results

SUMMARY_COLUMNS = ["section", "group", "category", "count", "total", "percent"]

FSM = "full-splice_match"
MAPPING_CATEGORIES = ["incomplete-splice_match", "novel_in_catalog", "novel_not_in_catalog"]

# Outcomes of an artifact, in the order shown in the report
RESCUED_OUTCOMES = ["rescued_reference", "rescued_lr_defined"]
EXCLUSION_REASONS = [
    "reference_already_represented",  # its reference transcript has an isoform passing the filter
    "mono_exonic_excluded",           # excluded by --rescue_mono_exonic
    "category_not_eligible",          # categories that rescue never considers (genic, antisense, ...)
    "mapping_not_run",                # ISM/NIC/NNC artifacts in automatic mode
    "no_mapping_hit",                 # candidate without any alignment to the targets
    "no_hit_passes_filter",           # all the targets it maps to fail the filter
]


def _row(section, group, category, count, total):
    percent = round(100 * count / total, 2) if total else 0.0
    return dict(section=section, group=group, category=category,
                count=int(count), total=int(total), percent=percent)


def _exon_class(exons):
    return "mono-exonic" if exons == 1 else "multi-exonic"


def hit_targets_table(hits_df, classif_df, reference_filter_file, strategy, thr=0.7,
                      reasons_file=None):
    """Filter results of the mapping hits (one row per candidate-target pair).

    Uses the same functions as rescue_by_mapping(), so the filter result of each
    target is the one used to rescue.

    Args:
        hits_df (pd.DataFrame): mapping hits (rescue_candidate, mapping_hit, ...)
        classif_df (pd.DataFrame): filter classification of the long-read transcriptome
        reference_filter_file (str): filter output for the reference transcriptome
            (rules classification or ML isoform predictions)
        strategy (str): 'rules' or 'ml'
        thr (float): ML probability threshold
        reasons_file (str): rules only, filtering reasons of the reference transcriptome

    Returns:
        pd.DataFrame: rescue_candidate, mapping_hit, hit_origin, hit_filter_result,
            plus hit_POS_MLprob (ml) or hit_filter_reason (rules)
    """
    combined = merge_classifications(reference_filter_file, classif_df, strategy, thr)
    hits = add_filter_results(hits_df, combined, classif_df)
    cols = ["rescue_candidate", "mapping_hit", "hit_origin", "hit_filter_result"]
    if strategy == "ml":
        cols.append("hit_POS_MLprob")
        hits = hits[cols]
    else:
        hits = hits[cols]
        reasons = pd.DataFrame(columns=["isoform", "filter_reason"])
        if reasons_file is not None:
            written = pd.read_csv(reasons_file, sep="\t")
            # With no artifacts the filter writes the classification header instead
            if "filter_reason" in written.columns:
                reasons = written[["isoform", "filter_reason"]]
        hits = hits.merge(reasons.rename(columns={"isoform": "mapping_hit",
                                                  "filter_reason": "hit_filter_reason"}),
                          on="mapping_hit", how="left")
        # Long-read targets are evaluated by the user's filter run, not by the reference one
        hits.loc[hits["hit_origin"] != "reference", "hit_filter_reason"] = pd.NA
    return hits


def artifact_outcomes(classif_df, rescue_df, mode, rescue_mono_exonic="all",
                      candidates=None, hits=None):
    """Assign an outcome to every artifact of the filter.

    Args:
        classif_df (pd.DataFrame): filter classification
        rescue_df (pd.DataFrame): rescue table (artifact, assigned_transcript,
            rescue_mode, origin, reintroduced)
        mode (str): rescue mode, 'automatic' or 'full'
        rescue_mono_exonic (str): value of --rescue_mono_exonic ('all', 'fsm' or 'none')
        candidates (list): rescue candidates (full mode)
        hits (pd.DataFrame): output of hit_targets_table() (full mode)

    Returns:
        pd.DataFrame: artifact, structural_category, exon_class, outcome,
            rescue_mode, origin
    """
    artifacts = classif_df.loc[classif_df["filter_result"] == "Artifact",
                               ["isoform", "structural_category", "exons"]]
    artifacts = artifacts.rename(columns={"isoform": "artifact"}).reset_index(drop=True)
    artifacts["exon_class"] = artifacts["exons"].map(_exon_class)

    # Rescued artifacts: one rescue mode and origin per artifact (ties share the origin)
    rescued = rescue_df.drop_duplicates("artifact")[["artifact", "rescue_mode", "origin"]]
    out = artifacts.merge(rescued, on="artifact", how="left")

    candidates = set(candidates) if candidates is not None else set()
    hit_any, hit_pass = set(), set()
    if hits is not None and not hits.empty:
        # Unmapped candidates are kept in the SAM file with no target
        hit_any = set(hits.loc[hits["mapping_hit"].notna(), "rescue_candidate"])
        hit_pass = set(hits.loc[hits["hit_filter_result"] == "Isoform", "rescue_candidate"])

    def outcome(r):
        if pd.notna(r["origin"]):
            return "rescued_reference" if r["origin"] == "reference" else "rescued_lr_defined"
        mono = r["exons"] == 1
        if r["structural_category"] == FSM:
            # Automatic rescue only reintroduces references with no FSM passing the filter
            if mono and rescue_mono_exonic == "none":
                return "mono_exonic_excluded"
            return "reference_already_represented"
        if r["structural_category"] not in MAPPING_CATEGORIES:
            return "category_not_eligible"
        if mode != "full":
            return "mapping_not_run"
        if r["artifact"] not in candidates:
            if mono and rescue_mono_exonic != "all":
                return "mono_exonic_excluded"
            # ISM whose reference transcript is represented
            return "reference_already_represented"
        if r["artifact"] not in hit_any:
            return "no_mapping_hit"
        if r["artifact"] not in hit_pass:
            return "no_hit_passes_filter"
        # Not expected: a candidate with a passing hit is always in the rescue table
        return "no_hit_passes_filter"

    out["outcome"] = out.apply(outcome, axis=1) if not out.empty else pd.Series(dtype=str)
    return out[["artifact", "structural_category", "exon_class", "outcome", "rescue_mode", "origin"]]


def gene_recovery(classif_df, rescue_df, inclusion_list, ref_genes=None):
    """Genes left without isoforms by the filter that the rescue recovers.

    A gene is lost when none of its isoforms passes the filter, and recovered when
    a reference transcript of the gene is added by the rescue. Novel genes cannot be
    recovered from the reference and are left out.

    Args:
        ref_genes (dict): reference transcript -> gene. When not given, the gene of the
            artifact that originated the added transcript is used.

    Returns:
        dict: genes, lost_genes, recovered_genes (all as sets)
    """
    genes = classif_df[["associated_gene", "filter_result"]].copy()
    genes = genes[~genes["associated_gene"].astype(str).str.startswith("novel")]
    passing = set(genes.loc[genes["filter_result"] == "Isoform", "associated_gene"])
    all_genes = set(genes["associated_gene"])
    lost = all_genes - passing

    added = set(inclusion_list)
    if ref_genes is None:
        gene_of_artifact = classif_df.set_index("isoform")["associated_gene"]
        rows = rescue_df[rescue_df["assigned_transcript"].isin(added)]
        added_genes = set(rows["artifact"].map(gene_of_artifact).dropna())
    else:
        added_genes = {ref_genes[t] for t in added if t in ref_genes}
    return dict(genes=all_genes, lost_genes=lost, recovered_genes=lost & added_genes)


def rules_failed_requisites(hits):
    """Count the requisites failed by the reference targets that do not pass the filter.

    A target is counted once per requisite, even if several candidates map to it.
    Reasons are written by the rules filter as '<column>: <value> ...' or
    'NA value in <column>'.
    """
    failed = hits[(hits["hit_origin"] == "reference") & (hits["hit_filter_result"] == "Artifact")]
    failed = failed[failed["mapping_hit"].notna()].drop_duplicates("mapping_hit")
    counts = {}
    for reasons in failed["hit_filter_reason"].dropna():
        requisites = set()
        for reason in str(reasons).split("; "):
            if reason.startswith("NA value in "):
                requisites.add(f"{reason[len('NA value in '):]} (NA)")
            elif reason:
                requisites.add(reason.split(":")[0])
        for req in requisites:
            counts[req] = counts.get(req, 0) + 1
    return counts, len(failed)


def summarize_rescue(classif_df, rescue_df, inclusion_list, mode, strategy=None,
                     rescue_mono_exonic="all", candidates=None, hits=None,
                     ref_genes=None, thr=None):
    """Build the long-format summary table of a rescue run.

    Returns:
        tuple: (summary DataFrame with SUMMARY_COLUMNS, artifact outcomes DataFrame)
    """
    if isinstance(inclusion_list, pd.DataFrame):
        # automatic rescue returns an empty DataFrame when there is nothing to rescue
        inclusion_list = inclusion_list["isoform"] if "isoform" in inclusion_list else []
    inclusion = pd.Series(list(inclusion_list), dtype=object).drop_duplicates()
    outcomes = artifact_outcomes(classif_df, rescue_df, mode, rescue_mono_exonic,
                                 candidates, hits)
    rows = []
    n_artifacts = len(outcomes)
    rescued = outcomes[outcomes["outcome"].isin(RESCUED_OUTCOMES)]
    assigned = rescue_df.drop_duplicates("assigned_transcript")
    n_considered = int((~outcomes["outcome"].isin(
        ["reference_already_represented", "mono_exonic_excluded",
         "category_not_eligible", "mapping_not_run"])).sum())

    # 1. Units of the rescue
    rows += [_row("overview", "artifacts", "artifacts", n_artifacts, n_artifacts),
             _row("overview", "artifacts", "considered_for_rescue", n_considered, n_artifacts),
             _row("overview", "artifacts", "rescued", len(rescued), n_artifacts),
             _row("overview", "transcripts", "assigned_transcripts", len(assigned), len(assigned)),
             _row("overview", "transcripts", "added_reference_transcripts", len(inclusion), len(assigned))]

    # 2. Rescue mode x origin, for rescued artifacts and for added transcripts
    by_artifact = rescue_df.drop_duplicates("artifact")
    for (rmode, origin), n in by_artifact.groupby(["rescue_mode", "origin"]).size().items():
        rows.append(_row("mode_origin_artifacts", rmode, origin, n, len(by_artifact)))
    added_rows = rescue_df[rescue_df["reintroduced"] == "yes"]
    for (rmode, origin), n in added_rows.groupby(["rescue_mode", "origin"]).size().items():
        rows.append(_row("mode_origin_added", rmode, origin, n, len(added_rows)))

    # 3. Outcome of the artifacts by their structural category (rescued and exclusion reasons)
    cat_totals = outcomes["structural_category"].value_counts()
    for (cat, outcome), n in outcomes.groupby(["structural_category", "outcome"]).size().items():
        rows.append(_row("artifact_outcome", cat, outcome, n, cat_totals[cat]))
    for outcome, n in outcomes["outcome"].value_counts().items():
        rows.append(_row("artifact_outcome_total", "all", outcome, n, n_artifacts))

    # 4. Transcriptome composition before and after the rescue. Added transcripts are
    #    labelled with the category of the artifact that reintroduced them.
    passing = classif_df.loc[classif_df["filter_result"] == "Isoform", "structural_category"]
    n_before = len(passing)
    n_after = n_before + len(inclusion)
    for cat, n in passing.value_counts().items():
        rows.append(_row("composition", "before", cat, n, n_before))
        rows.append(_row("composition", "after", cat, n, n_after))
    cat_of_artifact = classif_df.set_index("isoform")["structural_category"]
    first = added_rows.drop_duplicates("assigned_transcript")
    first = first[first["assigned_transcript"].isin(set(inclusion))]
    origin_cat = first["artifact"].map(cat_of_artifact).fillna("unknown")
    for cat, n in origin_cat.value_counts().items():
        rows.append(_row("composition", "after", f"rescued_from:{cat}", n, n_after))

    # 5. Gene-level recovery
    genes = gene_recovery(classif_df, rescue_df, inclusion, ref_genes)
    n_genes, n_lost = len(genes["genes"]), len(genes["lost_genes"])
    rows += [_row("gene_recovery", "genes", "genes_with_isoforms_after_filter", n_genes - n_lost, n_genes),
             _row("gene_recovery", "genes", "genes_lost_by_filter", n_lost, n_genes),
             _row("gene_recovery", "lost_genes", "recovered_by_rescue", len(genes["recovered_genes"]), n_lost)]

    # 6. Multiplicity: number of rescued artifacts that point to each assigned transcript
    pairs = rescue_df.drop_duplicates(["artifact", "assigned_transcript"])
    mult = pairs.groupby(["origin", "assigned_transcript"]).size()
    for origin, sizes in mult.groupby(level=0):
        dist = sizes.value_counts().sort_index()
        for k, n in dist.items():
            rows.append(_row("multiplicity", origin, str(k), n, len(sizes)))

    # 7. Mono- vs multi-exonic artifacts
    exon_totals = outcomes["exon_class"].value_counts()
    status = outcomes["outcome"].isin(RESCUED_OUTCOMES).map({True: "rescued", False: "not_rescued"})
    for (ec, st), n in outcomes.assign(status=status).groupby(["exon_class", "status"]).size().items():
        rows.append(_row("exon_class", ec, st, n, exon_totals[ec]))

    # 8. Filter diagnostics on the targets the candidates map to
    if hits is not None and not hits.empty:
        targets = hits[hits["mapping_hit"].notna()].drop_duplicates("mapping_hit")
        for (origin, res), n in targets.groupby(["hit_origin", "hit_filter_result"]).size().items():
            rows.append(_row("target_filter", origin, res, n, (targets["hit_origin"] == origin).sum()))
        if strategy == "rules" and "hit_filter_reason" in hits.columns:
            counts, n_failed = rules_failed_requisites(hits)
            for req, n in sorted(counts.items(), key=lambda x: -x[1]):
                rows.append(_row("failed_requisites", "reference_targets", req, n, n_failed))
        if strategy == "ml" and thr is not None:
            ref = targets[targets["hit_origin"] == "reference"]
            above = int((ref["hit_POS_MLprob"] >= thr).sum())
            rows += [_row("ml_threshold", "reference_targets", "above_threshold", above, len(ref)),
                     _row("ml_threshold", "reference_targets", "below_threshold", len(ref) - above, len(ref))]

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary, outcomes


def write_rescue_summary(prefix, summary, outcomes, hits=None, mode="automatic",
                         strategy=None, thr=None):
    """Write the summary tables read by the rescue report.

    Files:
        {prefix}_rescue_summary.tsv: long table (section, group, category, count, total, percent)
        {prefix}_rescue_artifact_outcomes.tsv: outcome of every artifact
        {prefix}_rescue_hit_targets.tsv: filter result of each target hit by a candidate (full mode)
    """
    files = {"summary": f"{prefix}_rescue_summary.tsv",
             "outcomes": f"{prefix}_rescue_artifact_outcomes.tsv"}
    meta = [_row("run", "mode", mode, 0, 0)]
    if strategy is not None and mode == "full":
        meta.append(_row("run", "strategy", strategy, 0, 0))
    if thr is not None and mode == "full" and strategy == "ml":
        meta.append(_row("run", "threshold", str(thr), 0, 0))
    pd.concat([pd.DataFrame(meta, columns=SUMMARY_COLUMNS), summary]).to_csv(
        files["summary"], sep="\t", index=False)
    outcomes.to_csv(files["outcomes"], sep="\t", index=False)
    if hits is not None:
        files["hits"] = f"{prefix}_rescue_hit_targets.tsv"
        hits[hits["mapping_hit"].notna()].drop_duplicates(["mapping_hit"]).drop(columns=["rescue_candidate"]).to_csv(
            files["hits"], sep="\t", index=False)
    return files
