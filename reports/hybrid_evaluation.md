# Hybrid Fraud System Evaluation

Untouched chronological test partition.

| System | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Review rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classifier only | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 0.0909 | 0.1111 |
| Hybrid flagged | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0222 |
| Hybrid high risk | N/A | 0.0000 | N/A | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Component ablation

| Ablation | Recall | FPR | PR-AUC | Review rate |
|---|---:|---:|---:|---:|
| without supervised | 0.0000 | 0.0909 | 0.0332 | 0.0889 |
| without anomaly | 1.0000 | 0.0000 | 1.0000 | 0.0222 |
| without behavioral | 1.0000 | 0.0000 | 1.0000 | 0.0222 |
| without graph | 1.0000 | 0.0000 | 1.0000 | 0.0222 |

## Limitations

Metrics use dataset source 'development_fixture' and do not establish production, fairness, or autonomous-decision readiness. PaySim is simulated transaction data and the lending context is deterministic synthetic enrichment.
