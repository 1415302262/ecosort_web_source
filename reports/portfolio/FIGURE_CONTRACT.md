# EcoSort cascade portfolio figure contract

- **Core conclusion:** Routing 15.6% of test images to a VLM raises paired test-set Macro-F1 by 5.13 percentage points while exposing a clear network-latency and label-boundary trade-off.
- **Figure archetype:** quantitative grid with one dominant paired-performance panel.
- **Target output:** recruiter-facing README/report figure; 180 mm × 145 mm; editable SVG and PDF plus 300 dpi PNG.
- **Backend:** Python (matplotlib only).
- **Panel a:** paired Local vs Cascade Accuracy, Macro-F1 and Macro-Recall.
- **Panel b:** error accounting from 80 local errors through 49 corrections and 4 introduced errors to 35 cascade errors.
- **Panel c:** per-class F1 delta with paired-bootstrap 95% confidence intervals.
- **Panel d:** empirical latency distributions for all 911 test samples, with P50/P95 annotations.
- **Hero evidence:** paired Macro-F1 delta +5.13 pp, bootstrap 95% CI +3.33 to +7.28 pp, exact McNemar p=7.05×10⁻¹¹.
- **Validation evidence:** 49 corrected local errors, 4 introduced errors, Net Corrected=45.
- **Trade-off evidence:** 142/911 VLM calls and cascade P95 latency 4.14 s.
- **Source data:** fixed leakage-excluded 911-sample `cascade_predictions.csv`, `cascade_metrics.json`, `cascade_robustness_metrics.json`, and `cascade_per_class_delta.csv`.
- **Image integrity:** no sample image is modified or included in this quantitative figure.
- **Reviewer risks:** one failed oversized request, one unmapped `trash` output, directory-level label noise, one provider/model version, and confidence intervals that do not cover deployment drift.
