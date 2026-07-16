# biotech_postings_4857_classified.csv

## What this is
4,857 biotech/pharma-relevant job postings pulled from the full raw LinkedIn Job Postings
2023-2024 archive (Kaggle: https://github.com/ArshKA/LinkedIn-Job-Scraper), filtered and
auto-classified. This is a scale-check sample that complements, not replaces, the
hand-labeled 836-posting dataset built with the TKS-style codebook.

## Source
- Original archive: 123,849 unique postings, all dated Dec 2023 - Apr 2024
  (entirely post-GenAI, per original_listed_time). No pre-Nov-2022 postings exist
  in this source, confirming the need for a separate pre-GenAI baseline (O*NET).

## Filtering
A posting was kept if EITHER:
- it was tagged with LinkedIn industry_id 12 (Biotechnology Research), 3238
  (Biotechnology), or 15 (Pharmaceutical Manufacturing), OR
- its title + description matched a biotech/pharma keyword regex (biotech, genomic,
  CRISPR, gene/cell therapy, bioinformatic, synthetic biology, biomanufacturing,
  molecular biology, life science, pharma, clinical trial, biopharma, proteomic,
  multi-omic, RNA therapeutic, mRNA, nucleic acid, drug discovery, biologics, GMP,
  genome editing)

## Columns
| Column | Description |
|---|---|
| job_id | LinkedIn posting ID |
| title, company_name, location, formatted_experience_level | Standard posting metadata |
| min_salary, med_salary, max_salary, normalized_salary, pay_period | Compensation fields as provided by source |
| remote_allowed | Source field, mostly null |
| taxonomy_category | Auto-assigned to the project's 7-category scientific-capability taxonomy (AI/ML & TechBio, Functional Genomics & Multi-omics, Cell/Gene & Genome Editing Therapies, Engineering Biology & Synthetic Biology, Computational Biology & Digital Biotech, RNA & Nucleic Acid Technologies, Advanced Biomanufacturing & Deep Biotech, or "General Biotech / Other" if no keyword matched) |
| wrc_category | Auto-assigned to the work-role-function categories used in the earlier slide (Biomanufacturing & process operations, Clinical development & medical affairs, Data science & computational biology, Technology & software engineering, Oversight/governance & leadership, Commercialization & business strategy, Research & discovery, or "Other / Unclassified") |
| ai_ml_mention | Boolean, True if title+description contains AI/ML terminology (machine learning, artificial intelligence, deep learning, generative AI, LLM, neural network, NLP, etc.) |
| python_mention, sas_mention | Boolean, simple tool-mention flags |
| ind_nda_hits | Count of "IND" / "NDA" regulatory-term occurrences in title+description |
| description | Full raw posting text |

## IMPORTANT: this is rule-based, not hand-labeled
Both `taxonomy_category` and `wrc_category` were assigned by regex/keyword rules on
title and description text, applied in a fixed priority order (first matching
category wins). This is NOT the TKS-style manual codebook used for the 836-posting
set. Expect noisier boundaries:
- ~65% of postings fall into "General Biotech / Other" on the 7-category taxonomy
  (keyword rules mostly catch technical/scientific-domain postings, not generic
  pharma/clinical/commercial roles)
- ~11% fall into "Other / Unclassified" on the WRC scheme
- Category assignment is single-label (first match), so postings with multiple
  overlapping skills are forced into one bucket

Treat this file as a starting point for expanding the labeled set or for validating
patterns found in the 836-posting analysis at larger scale, not as ground truth.

## Suggested next steps for the labeling phase
- Sample from `taxonomy_category == "General Biotech / Other"` to check whether the
  7-category keyword rules need expansion (current rule set is intentionally
  conservative)
- Cross-tabulate `taxonomy_category` x `wrc_category` to see where the two axes
  (scientific domain vs. job function) diverge or agree
- Use `ai_ml_mention`, `python_mention`, `ind_nda_hits` as candidate weak-label
  features for the AI-intensity score (0-3) in the TKS codebook
