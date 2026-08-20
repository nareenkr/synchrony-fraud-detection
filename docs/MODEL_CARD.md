# Model Card

## Intended use

Synchrony is a portfolio prototype for fraud decision support in a digital-lending workflow. It prioritizes applications for approval or human investigation. It is not validated, calibrated, secured, or governed for real lending decisions, and `HIGH_RISK` must not be treated as authority to deny credit.

## Data

The pipeline accepts PaySim-shaped transaction records as its public base data. PaySim is synthetic mobile-money transaction data, not lending-application ground truth. Lending, device, IP, login, bank-account, and application-history attributes are deterministic synthetic enrichment and are cataloged in `data/feature_provenance.yaml`.

The bundled development fixture has 600 rows and deliberately oversamples source-level fraud behavior so every chronological partition exercises threshold and metric code. Its synthetic lending/device enrichment never conditions on the fraud label. Metrics are retained for pipeline verification only; benchmark claims require the full public dataset and representative lending-fraud labels.

## Models and policy

- Class-weighted logistic regression baseline
- Imbalance-weighted XGBoost challenger
- Isolation Forest with empirical training-score calibration
- Transparent behavioral/velocity rules
- Shared-device, IP, and bank-account graph proxies
- Configurable weighted risk score and approval/manual-review/high-risk thresholds

Selection uses chronological validation data and emphasizes fraud recall subject to a false-positive/review-capacity constraint. The test partition is used once after model and threshold selection.

The retained fixture artifact selects XGBoost. On its untouched 90-row test partition it records recall 1.00, FPR 0.091, precision 0.20, F1 0.333, ROC-AUC 1.00, and PR-AUC 1.00. These optimistic ranking metrics reflect the designed fixture behavior and must not be presented as real lending performance.

## Explanations

Offline evaluation retains a SHAP explanation. Runtime explanations use SHAP for the selected XGBoost tree model and local probability perturbation for supported linear models. Numeric attributions never cross the public response boundary; users see curated reasons and safe signal categories.

## Responsible-AI limitations

No race, religion, or deliberately sensitive trait is used. Their absence does not prove fairness: geography, income, tenure, device access, and data completeness may act as proxies or create unequal error rates. The segment report uses only non-sensitive operational bands, includes support and uncertainty, and suppresses low-support interpretation.

Fraud labels may reflect past investigation practices and selective labeling. PaySim-to-lending domain shift, synthetic assumptions, concept drift, adversarial adaptation, and calibration remain material limitations. False positives can delay legitimate applicants, so human review, investigator context, audit trails, monitoring, and an appeal mechanism are required.

## Monitoring required before any real use

- Fraud recall, precision, false-positive rate, calibration, and review volume
- Segment error rates with adequate support and uncertainty
- Data quality, missingness, drift, and entity-link cardinality changes
- Explanation stability and investigator override outcomes
- Latency, dependency readiness, and state-store consistency
- Periodic privacy, security, model-risk, and policy review

## Adaptation boundary

The prototype does not retrain itself on its own predictions. Investigators supply verified
outcomes, and the learning monitor reports reviewed sample size, false-positive reviews, missed
fraud, and whether the configured evidence threshold warrants retraining review. A recommendation
must lead to an offline, reproducible training run, untouched-test and segment evaluation, formal
approval, and a new hashed artifact version. This boundary prevents feedback loops, poisoning, and
unreviewed model changes while still demonstrating how the system adapts to emerging fraud.
