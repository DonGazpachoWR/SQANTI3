#!/bin/bash Rscript
#
#-------------------------------------------------------------------------------
#                       _________________________
#
#                         SQANTI3 RESCUE REPORT
#                       _________________________
#
# Author: Ángeles Arzalluz-Luque
#
# Contact: angeles.arzalluz@gmail.com
#
# Affiliation: Institute for Integrative Systems Biology, CSIC, Valencia, Spain
#
# Last updated: October 2026
#
# The numbers are computed in Python (src/utilities/rescue/rescue_summary.py)
# and written to <prefix>_rescue_summary.tsv. This script only draws them.
#
#-------------------------------------------------------------------------------



####------------------------- RESCUE REPORT ARG LIST -----------------------####

#### Define script arguments ####

option_list <- list(
  optparse::make_option(c("-d","--dir"), type = "character",
                        help = "Output/input directory - must be same as SQ3 rescue."),
  optparse::make_option(c("-o","--output"), type = "character", default = "SQANTI3",
                        help = "Prefix of the rescue output files.")
)

# Parse arguments
opt_parser = optparse::OptionParser(option_list = option_list)
opt = optparse::parse_args(opt_parser)


message("\n--------------------------------------------------")
message("\n \t SQANTI3 rescue report")
message("\n--------------------------------------------------")

suppressPackageStartupMessages({
  require(ggplot2)
  require(gridExtra)
  require(grid)
})

####------------------------ INPUTS ---------------------------####

prefix <- file.path(opt$dir, opt$output)
summary_file <- paste0(prefix, "_rescue_summary.tsv")
hits_file <- paste0(prefix, "_rescue_hit_targets.tsv")
pdf_file <- paste0(prefix, "_SQANTI3_rescue_report.pdf")

if (!file.exists(summary_file)) {
  stop(paste("Rescue summary not found:", summary_file))
}
s <- read.table(summary_file, sep = "\t", header = TRUE, as.is = TRUE,
                quote = "", comment.char = "")
hits <- NULL
if (file.exists(hits_file)) {
  hits <- read.table(hits_file, sep = "\t", header = TRUE, as.is = TRUE,
                     quote = "", comment.char = "")
}

section <- function(name) s[s$section == name, , drop = FALSE]
run_value <- function(key) {
  x <- s$category[s$section == "run" & s$group == key]
  if (length(x) == 0) NA else x[1]
}
mode <- run_value("mode")
strategy <- run_value("strategy")
threshold <- suppressWarnings(as.numeric(run_value("threshold")))


####------------------------ PLOT THEME ---------------------------####

# Set theme parameters (from SQANTI3_report.R)
sq_theme <- theme_classic(base_family = "Helvetica") +
  theme(plot.title = element_text(lineheight=.4, size=15, hjust = 0.5)) +
  theme(plot.subtitle = element_text(size=10, hjust = 0.5, colour = "grey30")) +
  theme(plot.caption = element_text(size=9, hjust = 0, colour = "grey30")) +
  theme(plot.margin = unit(c(1.5,1,1,1), "cm")) +
  theme(axis.line.x = element_line(color="black", linewidth = 0.4),
        axis.line.y = element_line(color="black", linewidth = 0.4)) +
  theme(axis.title.x = element_text(size=13),
        axis.text.x  = element_text(size=11),
        axis.title.y = element_text(size=13),
        axis.text.y  = element_text(vjust=0.5, size=11) ) +
  theme(legend.text = element_text(size = 10),
        legend.title = element_text(size=11),
        legend.key.size = unit(0.5, "cm"))

theme_set(sq_theme)

# Structural categories: same labels and colours as the QC report
cat.labels <- c(`full-splice_match` = "FSM", `incomplete-splice_match` = "ISM",
                novel_in_catalog = "NIC", novel_not_in_catalog = "NNC",
                genic = "Genic\nGenomic", antisense = "Antisense", fusion = "Fusion",
                intergenic = "Intergenic", genic_intron = "Genic\nIntron")
cat.palette <- c(FSM="#6BAED6", ISM="#FC8D59", NIC="#78C679",
                 NNC="#EE6A50", `Genic\nGenomic`="#969696", Antisense="#66C2A4",
                 Fusion="goldenrod1", Intergenic = "darksalmon", `Genic\nIntron`="#41B6C4",
                 unknown = "grey80")
short_cat <- function(x) {
  out <- unname(cat.labels[x])
  out[is.na(out)] <- x[is.na(out)]
  factor(out, levels = unique(c(unname(cat.labels), "unknown", out)))
}

# Rescue outcomes, in a fixed order (rescued first, then exclusion reasons)
outcome.labels <- c(rescued_reference = "Rescued: reference",
                    rescued_lr_defined = "Rescued: long-read defined",
                    reference_already_represented = "Reference already represented",
                    mono_exonic_excluded = "Mono-exonic excluded",
                    category_not_eligible = "Category not eligible",
                    mapping_not_run = "Mapping not run (automatic mode)",
                    no_mapping_hit = "No mapping hit",
                    no_hit_passes_filter = "No target passes the filter",
                    failed_evidence_check = "Targets fail the evidence check")
# Categorical slots in fixed order; the structural exclusion is a neutral grey
outcome.palette <- setNames(c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9e9e9e",
                              "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
                            unname(outcome.labels))
origin.palette <- c(reference = "#2a78d6", lr_defined = "#eb6834")
short_outcome <- function(x) factor(unname(outcome.labels[x]), levels = unname(outcome.labels))

pct <- function(x) paste0(format(round(x, 1), nsmall = 1), "%")

empty_page <- function(title, text) {
  ggplot() + annotate("text", x = 0, y = 0, label = text, size = 4.5) +
    labs(title = title) + theme_void() +
    theme(plot.title = element_text(size = 15, hjust = 0.5))
}

table_page <- function(title, df, subtitle = NULL) {
  tg <- gridExtra::tableGrob(df, rows = NULL,
                             theme = gridExtra::ttheme_minimal(base_size = 11))
  heading <- textGrob(title, gp = gpar(fontsize = 15))
  if (is.null(subtitle)) {
    grid.arrange(heading, tg, ncol = 1, heights = c(0.12, 0.88))
  } else {
    sub <- textGrob(subtitle, gp = gpar(fontsize = 10, col = "grey30"))
    grid.arrange(heading, sub, tg, ncol = 1, heights = c(0.08, 0.08, 0.84))
  }
}


####------------------------ REPORT ---------------------------####

pdf(file = pdf_file, width = 9, height = 7.5)

#### Page 1: overview ####
ov <- section("overview")
ov_names <- c(artifacts = "Artifacts after the filter",
              considered_for_rescue = "Artifacts considered for rescue",
              rescued = "Artifacts rescued",
              assigned_transcripts = "Transcripts assigned to rescued artifacts",
              added_reference_transcripts = "Reference transcripts added")
ov_den <- c(artifacts = "", considered_for_rescue = "of artifacts",
            rescued = "of artifacts", assigned_transcripts = "",
            added_reference_transcripts = "of assigned transcripts")
ov_tab <- data.frame(
  Measure = unname(ov_names[ov$category]),
  Count = ov$count,
  Percent = ifelse(ov_den[ov$category] == "", "", paste(pct(ov$percent), ov_den[ov$category])),
  check.names = FALSE)
subtitle <- paste0("Rescue mode: ", mode,
                   ifelse(is.na(strategy), "", paste0("  |  Strategy: ", strategy)),
                   ifelse(is.na(threshold), "", paste0("  |  ML threshold: ", threshold)))
table_page("SQANTI3 rescue summary", ov_tab, subtitle)

#### Page 2: rescue mode x origin ####
mo <- section("mode_origin_artifacts")
ma <- section("mode_origin_added")
if (nrow(mo) > 0) {
  tab <- merge(mo[, c("group", "category", "count", "percent")],
               ma[, c("group", "category", "count")],
               by = c("group", "category"), all = TRUE, suffixes = c("_art", "_add"))
  tab$count_add[is.na(tab$count_add)] <- 0
  tab_out <- data.frame(`Rescue mode` = tab$group, Origin = tab$category,
                        `Rescued artifacts` = tab$count_art,
                        `% of rescued artifacts` = pct(tab$percent),
                        `Transcripts added` = tab$count_add,
                        check.names = FALSE)
  table_page("Rescued artifacts by rescue mode and origin of the target", tab_out,
             "Automatic rescue only acts on FSM artifacts and only adds reference transcripts.")

  p <- ggplot(mo, aes(x = group, y = count, fill = category)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.5) +
    scale_fill_manual(values = origin.palette, name = "Origin of the target") +
    labs(x = "Rescue mode", y = "Rescued artifacts",
         title = "Rescued artifacts by rescue mode and origin",
         caption = "Empty combinations are structural: automatic rescue only assigns reference transcripts.")
  print(p)
} else {
  print(empty_page("Rescued artifacts by rescue mode and origin", "No artifacts were rescued."))
}

#### Page 3: rescued artifacts by structural category of the artifact ####
ao <- section("artifact_outcome")
resc <- ao[ao$category %in% c("rescued_reference", "rescued_lr_defined"), ]
if (nrow(resc) > 0) {
  resc$cat <- short_cat(resc$group)
  resc$origin <- sub("^rescued_", "", resc$category)
  lab <- aggregate(cbind(count, percent) ~ cat, data = resc, FUN = sum)
  p <- ggplot(resc, aes(x = cat, y = count, fill = origin)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.5) +
    geom_text(data = lab, aes(x = cat, y = count, label = pct(percent)),
              inherit.aes = FALSE, vjust = -0.4, size = 3.5) +
    scale_fill_manual(values = origin.palette, name = "Origin of the target") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
    labs(x = "Structural category of the artifact", y = "Rescued artifacts",
         title = "Rescued artifacts by structural category",
         subtitle = "Labels: percentage of the artifacts of each category that are rescued")
  print(p)
}

#### Page 4: transcriptome composition before and after the rescue ####
comp <- section("composition")
if (nrow(comp) > 0) {
  comp$added <- grepl("^rescued_from:", comp$category)
  comp$cat <- short_cat(sub("^rescued_from:", "", comp$category))
  comp$status <- ifelse(comp$added, "Added by the rescue (category of the artifact)",
                        "Passed the filter")
  comp$group <- factor(comp$group, levels = c("before", "after"),
                       labels = c("Before rescue", "After rescue"))
  # Stack the transcripts that passed the filter at the bottom and the added ones on top
  comp$stack <- factor(paste(comp$added, comp$cat),
                       levels = c(paste(TRUE, rev(levels(comp$cat))),
                                  paste(FALSE, rev(levels(comp$cat)))))
  p <- ggplot(comp, aes(x = group, y = count, fill = cat, alpha = status, group = stack)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = cat.palette, name = "Structural category") +
    scale_alpha_manual(values = c("Passed the filter" = 1,
                                  "Added by the rescue (category of the artifact)" = 0.45),
                       name = NULL) +
    labs(x = NULL, y = "Transcripts",
         title = "Transcriptome composition before and after the rescue",
         caption = paste("Added transcripts are reference transcripts; they are coloured by the category",
                         "of the artifact that reintroduced them.", sep = "\n"))
  print(p)
}

#### Page 5: outcome of all artifacts (rescued vs exclusion reasons) ####
if (nrow(ao) > 0) {
  ao$cat <- short_cat(ao$group)
  ao$outcome <- short_outcome(ao$category)
  p <- ggplot(ao, aes(x = cat, y = count, fill = outcome)) +
    geom_col(width = 0.6, position = "fill", colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = outcome.palette, name = "Outcome", drop = TRUE) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = "Structural category of the artifact", y = "Artifacts (%)",
         title = "Rescued and not rescued artifacts",
         subtitle = "Not rescued artifacts are split by the reason why they were excluded")
  print(p)

  #### Page 6: heatmap category x outcome ####
  p <- ggplot(ao, aes(x = outcome, y = cat, fill = percent)) +
    geom_tile(colour = "white", linewidth = 1) +
    geom_text(aes(label = count), size = 3.5,
              colour = ifelse(ao$percent > 60, "white", "black")) +
    scale_fill_gradient(low = "#e3eefb", high = "#0b3f86", limits = c(0, 100),
                        name = "% of the\ncategory") +
    labs(x = "Outcome", y = "Structural category of the artifact",
         title = "Where the artifacts of each category end up",
         subtitle = "Labels: number of artifacts") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
          axis.line = element_blank())
  print(p)
}

#### Evidence check of reference targets ####
ev <- section("evidence_check")
if (nrow(ev) > 0) {
  ev_names <- c(pass = "Reference targets passing the count requisites",
                failed = "Reference targets failing them (not reintroduced)",
                reassigned_to_lr_defined = "Their artifacts reassigned to a long-read isoform",
                sent_to_gene_residual = "Their artifacts sent to the gene residual")
  etab <- data.frame(Measure = unname(ev_names[ev$category]), Count = ev$count,
                     Percent = pct(ev$percent))
  table_page("Evidence check of reference targets", etab,
             "Count requisites of the rules applied to the counts aggregated from the artifacts.")
}

#### Page 7: gene-level recovery ####
gr <- section("gene_recovery")
if (nrow(gr) > 0) {
  gtab <- data.frame(
    Measure = c("Genes in the filter classification",
                "Genes with at least one isoform after the filter",
                "Genes lost by the filter (no isoform passes)",
                "Lost genes recovered by the rescue"),
    Count = c(gr$total[gr$category == "genes_lost_by_filter"],
              gr$count[gr$category == "genes_with_isoforms_after_filter"],
              gr$count[gr$category == "genes_lost_by_filter"],
              gr$count[gr$category == "recovered_by_rescue"]),
    Percent = c("", pct(gr$percent[gr$category == "genes_with_isoforms_after_filter"]),
                pct(gr$percent[gr$category == "genes_lost_by_filter"]),
                paste(pct(gr$percent[gr$category == "recovered_by_rescue"]), "of lost genes")))
  table_page("Gene-level recovery", gtab,
             "Novel genes are not counted, since the reference cannot recover them.")
}

#### Page 8: multiplicity ####
mu <- section("multiplicity")
if (nrow(mu) > 0) {
  mu$k <- as.integer(mu$category)
  mu$bin <- ifelse(mu$k >= 10, "10+", as.character(mu$k))
  mu <- aggregate(count ~ group + bin, data = mu, FUN = sum)
  mu$bin <- factor(mu$bin, levels = c(as.character(1:9), "10+"))
  p <- ggplot(mu, aes(x = bin, y = count, fill = group)) +
    geom_col(width = 0.7, position = position_dodge2(preserve = "single", padding = 0.15)) +
    scale_fill_manual(values = origin.palette, name = "Origin of the target") +
    labs(x = "Rescued artifacts assigned to the transcript", y = "Assigned transcripts",
         title = "How many artifacts point to each assigned transcript")
  print(p)
}

#### Page 9: mono- vs multi-exonic artifacts ####
ex <- section("exon_class")
if (nrow(ex) > 0) {
  ex$category <- factor(ex$category, levels = c("not_rescued", "rescued"),
                        labels = c("Not rescued", "Rescued"))
  p <- ggplot(ex, aes(x = group, y = count, fill = category)) +
    geom_col(width = 0.6, colour = "white", linewidth = 0.5) +
    geom_text(aes(label = pct(percent)), position = position_stack(vjust = 0.5), size = 3.5) +
    scale_fill_manual(values = c("Not rescued" = "grey80", "Rescued" = "#2a78d6"), name = NULL) +
    labs(x = NULL, y = "Artifacts", title = "Mono- and multi-exonic artifacts",
         caption = "Mono-exonic artifacts are handled according to --rescue_mono_exonic.")
  print(p)
}

#### Page 10: filter diagnostics on the targets ####
tf <- section("target_filter")
if (nrow(tf) > 0) {
  ttab <- data.frame(Origin = tf$group, `Filter result` = tf$category, Targets = tf$count,
                     Percent = pct(tf$percent), check.names = FALSE)
  table_page("Filter result of the targets hit by the candidates", ttab)
}
fr <- section("failed_requisites")
if (identical(strategy, "rules") && nrow(fr) > 0) {
  fr$category <- factor(fr$category, levels = rev(fr$category))
  p <- ggplot(fr, aes(x = category, y = count)) +
    geom_col(width = 0.6, fill = "#2a78d6") +
    geom_text(aes(label = pct(percent)), hjust = -0.2, size = 3.5) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "Requisite", y = "Reference targets failing it",
         title = "Requisites failed by the reference targets",
         subtitle = "Reference targets hit by a candidate that do not pass the rules",
         caption = "A target can fail several requisites. '(NA)' marks missing values.")
  print(p)
}
if (identical(strategy, "ml") && !is.null(hits) && "hit_POS_MLprob" %in% colnames(hits)) {
  ref_hits <- hits[hits$hit_origin == "reference" & !is.na(hits$hit_POS_MLprob), ]
  if (nrow(ref_hits) > 0) {
    p <- ggplot(ref_hits, aes(x = hit_POS_MLprob)) +
      geom_histogram(binwidth = 0.025, boundary = 0, fill = "#2a78d6", colour = "white") +
      geom_vline(xintercept = threshold, linetype = "dashed") +
      annotate("text", x = threshold, y = Inf, label = paste("threshold =", threshold),
               hjust = -0.1, vjust = 1.5, size = 3.5) +
      scale_x_continuous(limits = c(0, 1)) +
      labs(x = "Probability of being an isoform (POS_MLprob)", y = "Reference targets",
           title = "ML probability of the reference targets",
           subtitle = "Reference targets hit by a candidate; only those above the threshold can rescue")
    print(p)
  }
}

invisible(dev.off())
message(paste("Rescue report written to", pdf_file))
