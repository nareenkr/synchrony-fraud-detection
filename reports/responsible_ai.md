# Responsible-AI Segment Review

This prototype analysis uses non-sensitive operational segments only. It is not a legal fairness assessment and cannot establish production suitability.

| Dimension | Segment | Support | Fraud cases | Recall | FPR | Review rate | Note |
|---|---|---:|---:|---:|---:|---:|---|
| account_tenure_band | 181-365d | 2 | 0 | N/A | 0.000 | 0.000 | low support; interpretation suppressed |
| account_tenure_band | 366d+ | 86 | 1 | 1.000 | 0.000 | 0.012 |  |
| account_tenure_band | 8-30d | 2 | 1 | 1.000 | 0.000 | 0.500 | low support; interpretation suppressed |
| requested_amount_tertile | lower | 30 | 0 | N/A | 0.000 | 0.000 |  |
| requested_amount_tertile | middle | 30 | 2 | 1.000 | 0.000 | 0.067 |  |
| requested_amount_tertile | upper | 30 | 0 | N/A | 0.000 | 0.000 |  |
| loan_to_income_band | 0.25-0.75 | 38 | 2 | 1.000 | 0.000 | 0.053 |  |
| loan_to_income_band | 0.75-1.5 | 7 | 0 | N/A | 0.000 | 0.000 |  |
| loan_to_income_band | <=0.25 | 44 | 0 | N/A | 0.000 | 0.000 |  |
| loan_to_income_band | >1.5 | 1 | 0 | N/A | 0.000 | 0.000 | low support; interpretation suppressed |
| data_completeness | complete | 90 | 2 | 1.000 | 0.000 | 0.022 |  |

## Interpretation boundaries

- Race, religion, gender, and other protected traits are not model inputs and are not present in the prepared `development_fixture` data.
- Geography is omitted from segmentation because it may proxy protected status.
- Synthetic lending attributes and sparse fraud labels make disparities unstable; confidence intervals and support counts must accompany every rate.
- Label bias, domain shift, concept drift, and selective investigation labels remain unmeasured risks.
- False positives can delay access to credit. The manual-review band, investigator context, and an appeal path are required safeguards.
- HIGH_RISK is an investigation priority, not authority for autonomous denial.
