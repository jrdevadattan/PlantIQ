# π0 — Pharmaceutical Intelligence Operations Platform

### *Every prediction sourced. Every decision stored. Every operator informed. Every batch understood.*

> **Track A — Predictive Modelling** | Pharmaceutical Tablet Manufacturing | AI-Driven Manufacturing Intelligence

---

## What π0 Is

π0 is an AI-powered batch intelligence system built for pharmaceutical tablet manufacturing. Before a batch begins, an operator sets seven process parameters — granulation time, binder amount, drying temperature, drying time, compression force, machine speed, and lubricant concentration. π0 takes those seven numbers and simultaneously predicts five outcomes: tablet hardness, friability, dissolution rate, content uniformity, and energy consumption in kilowatt-hours.

It does not stop at prediction. As the batch runs, π0 monitors the machine's power consumption curve in real time, detects when something is deviating from the golden signature of a healthy batch, and fires a structured alert that tells the operator exactly what is wrong, which machine to walk to, and what physical action to take. Every prediction is explained in plain language with specific attribution to individual parameters. Every decision, every alert, every operator acknowledgement, and every model version is stored permanently and is retrievable in full at any future point.

π0 is designed to be read and used by three completely different people simultaneously — the floor operator who needs three numbers and one colour, the engineer who needs SHAP waterfall charts and confidence intervals, and the plant manager who needs cost in rupees and a weekly energy trend.

---

## The Problem π0 Solves

Pharmaceutical tablet manufacturing runs in batches. Each batch takes between 3 and 4 hours from preparation to quality testing. Current manufacturing systems evaluate batch outcomes only after the batch completes — meaning every problem is discovered after it has already happened, after the energy has already been wasted, after the tablets have already been made incorrectly.

**Batch-level variability** is the first problem. Energy usage and product quality fluctuate significantly across batches because of varying machine settings, material characteristics, and operator decisions. There is currently no way to predict, before a batch starts, whether the settings chosen will produce quality tablets at reasonable energy cost.

**Post-process discovery** is the second problem. When a batch fails quality control, the energy has already been consumed, the materials have already been used, and the production time has already been lost. There is no mechanism to intervene while the batch is running.

**Invisible explanations** are the third problem. Even when predictions exist, they are black boxes. An operator who is told "energy will be high" cannot act on that information without knowing which parameter is causing the high energy and what to change.

**Broken feedback** is the fourth problem. Models trained on historical data become less accurate over time as machine conditions change, materials shift, and processes evolve. Without a mechanism to detect and correct this drift, the system becomes silently wrong.

π0 addresses all four problems through its five-layer architecture.

---

## Table of Contents

1. [The Four Rules That Govern Every Decision](#1-the-four-rules)
2. [The Complete Data Journey](#2-the-complete-data-journey)
3. [System Architecture — Five Layers](#3-system-architecture)
4. [Layer 1 — Data Foundation](#4-layer-1--data-foundation)
5. [Layer 2 — The Intelligence Core](#5-layer-2--the-intelligence-core)
6. [Layer 3 — The Decision Engine](#6-layer-3--the-decision-engine)
7. [Layer 4 — The API Bridge](#7-layer-4--the-api-bridge)
8. [Layer 5 — The Three Dashboards](#8-layer-5--the-three-dashboards)
9. [The Feedback Loop](#9-the-feedback-loop)
10. [The Audit Architecture](#10-the-audit-architecture)
11. [The Alert and Acknowledgement System](#11-the-alert-and-acknowledgement-system)
12. [The Golden Batch System](#12-the-golden-batch-system)
13. [Technology Stack — Explained from First Principles](#13-technology-stack)
14. [Hackathon Build vs Production Architecture](#14-hackathon-vs-production)
15. [Gap Audit — What π0 Cannot Do and Why](#15-gap-audit)
16. [Future Scope — Eight Features with Technical Path](#16-future-scope)
17. [The One Question π0 Can Always Answer](#17-the-one-question)

---

## 1. The Four Rules

Every architectural decision in π0 was made by applying four rules. Knowing these rules explains why every component exists and why it is built the way it is.

---

**Rule 1 — Data Has a Birthplace**

Every number in π0 knows where it came from. A prediction does not simply say "hardness: 91.4 Newton." It says: that prediction was produced by model version 3.1.2, trained on 287 batches, using these ten input features, at this timestamp, with this SHAP breakdown showing which feature contributed how much. No number exists without a provenance record.

This rule exists because when something goes wrong — a batch fails, a prediction is way off, an auditor asks questions — you need to be able to trace the number back to its origin without ambiguity. A system where numbers appear without traceable sources is a system you cannot trust and cannot fix.

---

**Rule 2 — Every Component Has One Job**

The prediction model predicts. It does not alert. The alert engine fires alerts. It does not store them. The storage layer stores. It does not display. The dashboard displays. It does not compute. This principle is called Single Responsibility.

The reason it matters is debuggability. When something breaks in π0, you know exactly which component broke. You fix that one thing without touching anything else. A component that does two jobs fails in two different ways and breaks two other components when it does.

---

**Rule 3 — Design for Failure First**

π0 assumes every component will fail. The sensor feed drops. The model server crashes. The operator is away from the screen when an alert fires. The database write fails mid-record. Every component in π0 has a defined answer to the question: what happens to the rest of the system when I fail?

A system that only works when everything works is not designed for a real factory. A real factory has network interruptions, power fluctuations, machine restarts, and human error. π0 accounts for all of these explicitly.

---

**Rule 4 — The Loop Must Close**

A prediction system that does not learn from its predictions is not intelligent. It is a calculator with a shelf life. π0 closes the loop: predictions are made, batches complete, actual quality measurements are recorded, prediction errors are computed, model accuracy is tracked, and models are retrained when accuracy degrades. This loop is an architectural requirement, not an optional feature.

---

## 2. The Complete Data Journey

Before describing individual components, it is essential to understand the complete path a piece of data travels from the moment an operator enters batch parameters to the moment a decision is permanently stored.

Every box in this journey is a separate component. Every arrow is a defined data contract — a formal agreement about exactly what format data arrives in and exactly what format it leaves in. Nothing is assumed. Nothing is passed informally.

The journey begins when an operator enters seven process parameters. The **Input Validator** receives those seven numbers and applies four sequential checks: are all fields present, are they numbers, are they within physical machine limits, and are they within the model's training range. The first three failures stop the pipeline with a specific error message. The fourth is a warning — prediction proceeds with reduced confidence attached.

The validated parameters pass to the **Feature Engineer**, which computes three derived features from the seven inputs, producing a ten-feature vector where every derived value is tagged with its formula.

The ten features enter the **Multi-Target Predictor**, which runs five models simultaneously and produces five predictions, each with a confidence score and a model version record.

The predictions pass to the **SHAP Explainer**, which produces a ranked, quantified plain-English explanation of which parameter drove each output and by how much.

The explained predictions pass to the **Cost Translator**, which converts the energy prediction into rupees and projects monthly cost at the current rate.

The complete record is written permanently to the **Prediction Store** — immutable after the batch closes.

The batch then runs. The **Anomaly Detector** reads live power data every 30 seconds. The **Sliding Window Forecaster** updates the energy forecast continuously. If either exceeds a threshold, the **Alert Engine** fires a structured alert. Every alert lifecycle event is stored by the **Acknowledgement System**.

After the batch, QC enters actual measurements. The **Outcome Recorder** computes prediction error and updates the **Drift Detector's** rolling accuracy. If accuracy has degraded, a retraining flag is raised.

---

## 3. System Architecture

π0 is organised in five layers. Data flows downward from Layer 5, where the operator interacts, through Layers 4 and 3 where decisions are made, into Layer 2 where intelligence is applied, and down to Layer 1 where data is stored and managed.

```
┌══════════════════════════════════════════════════════════════════╗
║  LAYER 5 — USER LAYER                                            ║
║  Three dashboards: Operator View · Technical View · Manager View ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 4 — API BRIDGE LAYER                                      ║
║  FastAPI: the translator between Python models and browser UI    ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 3 — DECISION ENGINE LAYER                                 ║
║  Cost Translator · Alert Engine · Acknowledgement System         ║
║  Golden Batch Manager · Recommendation Engine                    ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 2 — INTELLIGENCE CORE LAYER                               ║
║  Multi-Target Predictor · SHAP Explainer · Anomaly Detector      ║
║  Sliding Window Forecaster · Confidence Scorer                   ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 1 — DATA FOUNDATION LAYER                                 ║
║  Input Validator · Feature Engineer · Prediction Store           ║
║  Model Registry · Feedback Loop Engine · Audit Store             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 4. Layer 1 — Data Foundation

Layer 1 is invisible to the operator. It handles everything that happens before intelligence is applied to data and everything that happens after a decision is made.

---

### Component 1.1 — The Input Validator

**One job:** Accept a set of batch parameters and either pass them as clean or reject them with a specific, actionable error message.

The validator applies four checks in sequence. Presence confirms all seven required fields exist. Type confirms every value is a number. Physical bounds confirm every value falls within the possible operating range of the machine — drying temperature, for example, cannot be below 35°C or above 85°C because those are the dryer's physical limits. The training distribution check confirms whether the value falls within what the model was trained on. The first three failures are hard stops. The fourth is a warning that reduces the prediction's confidence score.

This distinction is architecturally important. A physically impossible value indicates a data entry error and must be stopped. A value outside the training range may be a legitimate new operating condition — the system allows it to proceed but communicates honestly that its confidence for that setting is lower.

---

### Component 1.2 — The Feature Engineer

**One job:** Transform seven validated raw inputs into ten features by computing three derived values, each tagged with its formula and the source values used.

The three derived features capture physical relationships that the raw inputs cannot express individually. Temperature deviation captures how far drying temperature sits from the historical mean, because deviation from centre is more predictive than absolute temperature. Compression intensity captures the interaction between compression force and machine speed, because the same force at different speeds produces tablets with different mechanical properties. Drying efficiency ratio captures the relationship between temperature and time in the drying phase, because high temperature for short time and low temperature for long time have different effects on final moisture content.

Every derived feature is stored alongside its formula and the exact values used. Six months later, if a prediction is questioned, every input can be traced to its exact computation.

---

### Component 1.3 — The Prediction Store

**One job:** Write every prediction permanently as an immutable record with complete provenance, and allow actual QC outcomes to be appended after the batch completes.

A complete record contains the batch identity and timestamp, the model version and training data fingerprint, all ten input features, all five predictions with confidence scores, the complete SHAP breakdown across all five targets, the cost translation in rupees, any distribution warnings that were triggered, and placeholder fields for the actual QC outcomes that fill when QC completes their measurements.

The record is immutable after the batch is marked closed. If a correction is ever needed, a correction record is created that references the original. The original is never modified. This is called append-only architecture, and it is the foundation of every pharmaceutical audit system.

---

### Component 1.4 — The Model Registry

**One job:** Store every version of every model with complete metadata so that any historical prediction can be traced to the exact model that produced it.

The registry stores the model version identifier, creation date, deployment date, retirement date, current status, the training data it was built on with row counts for real versus augmented data, performance metrics at training time for all five targets, and performance over its deployment lifetime including running average prediction error. Every prediction record in the Prediction Store contains the model version identifier. The registry turns that identifier into a complete description of how the model was built.

---

### Component 1.5 — The Feedback Loop Engine

**One job:** Monitor the accuracy of predictions over time and flag when model performance has degraded to a point requiring retraining.

The engine maintains a 30-day rolling Mean Absolute Percentage Error for each of the five prediction targets. After every batch outcome is recorded, the engine recomputes the rolling accuracy for all five targets. If any target exceeds ten percent MAPE, an alert goes to the data team. If degradation is sustained over ten consecutive batches, the model is flagged for retraining.

---

## 5. Layer 2 — The Intelligence Core

Five components. Each does exactly one thing. All outputs flow into Layer 3 for decision-making.

---

### Component 2.1 — The Multi-Target Predictor

**One job:** Take ten features and produce five simultaneous predictions, each with a confidence score.

The predictor runs five XGBoost models simultaneously through a single interface. Each model is trained independently to predict one target. Five separate models rather than one combined model exists because the factors driving hardness are not the same as the factors driving energy. Compression force dominates hardness prediction. Drying time and drying temperature dominate energy prediction. A single model optimising for all five simultaneously would make compromises that reduce accuracy on every target. Five focused models each achieve better accuracy on their individual target.

Confidence is computed by comparing the incoming input vector to the statistical distribution of the training data. Inputs well inside the historical range receive high confidence above 0.85. Inputs at the edge receive medium confidence between 0.60 and 0.85. Inputs outside the training range receive low confidence below 0.60 and trigger a distribution warning that is stored with the prediction.

---

### Component 2.2 — The SHAP Explainer

**One job:** Take a completed prediction and produce a ranked, quantified, plain-language explanation of which parameter drove each output and by how much.

SHAP — SHapley Additive exPlanations — distributes a prediction across its contributing features using a mathematical method from cooperative game theory. The calculation determines how much of the deviation from the average prediction each of the ten features is responsible for, in the same way you might split a restaurant bill fairly among diners based on what each ordered.

The explainer runs a TreeExplainer, which is a highly efficient algorithm built specifically for tree-based models like XGBoost. It produces fifty values per prediction — one for each combination of five targets and ten features. These values are translated into plain English by a formatting layer: compression force added 3.8 kilowatt-hours because it was set higher than usual; drying time saved 2.1 kilowatt-hours because it was shorter than usual.

The SHAP output serves two purposes. First, it communicates to the operator why the system predicted what it predicted. Second, its output directly feeds the Recommendation Engine — the parameter with the highest positive SHAP contribution to energy overrun is the first parameter targeted for adjustment.

---

### Component 2.3 — The Power Curve Anomaly Detector

**One job:** Read the live power consumption curve during a running batch and produce an anomaly score every 30 seconds. When the score crosses a threshold, classify the fault type and generate a recommendation.

The anomaly detector uses an LSTM Autoencoder — a neural network that learned the shape of a normal pharmaceutical manufacturing power curve during training. During training, the model read examples of normal power curves and learned to compress them into a compact representation and then reconstruct them accurately. When it is good at this, it has learned what normal looks like.

During a live batch, the model attempts to reconstruct the current power curve using that learned normal pattern. If the curve is normal, reconstruction is accurate. If the curve is deviating — bearing wear, wet material, calibration drift — reconstruction is inaccurate, and that reconstruction error is the anomaly score.

The power curve has eight phases with distinct signatures. Preparation is idle at approximately 2 kilowatts. Milling has the highest vibration at approximately 8 millimetres per second and 36 kilowatts. Compression has the highest sustained power at approximately 45 kilowatts for 52 minutes, consuming 50 percent of the entire batch's energy.

Three fault patterns are identifiable from the curve:

| Fault | Where in Curve | Physical Cause | Recommended Action |
|---|---|---|---|
| Bearing Wear | Milling — gradually rising vibration across batches | Ball bearing surface wearing smooth, motor compensating | Schedule maintenance within 5 batches |
| Wet Raw Material | Drying — irregular power spikes instead of steady plateau | Moisture unevenly distributed in granules | Extend drying phase 3–4 minutes |
| Calibration Needed | Compression — uniformly elevated baseline for full 52 minutes | Compression die worn, more energy needed for same force | Schedule calibration at next changeover |

---

### Component 2.4 — The Sliding Window Forecaster

**One job:** Update the energy prediction every 30 seconds during a running batch using actual consumption data accumulated so far, and produce a progressively narrowing confidence interval.

The forecaster blends two signals. The first is the model's initial prediction based on parameters set before the batch. The second is an extrapolation from actual energy consumed so far. Early in the batch, the blend weights heavily toward the model prediction because little actual data exists. As the batch progresses and actual data accumulates, the blend shifts toward actual data. By the final quarter, the forecast is predominantly based on what has actually happened.

The confidence interval narrows as the batch progresses. At batch start, the interval might span plus or minus nine kilowatt-hours. By 80 percent completion, it narrows to plus or minus one kilowatt-hour. The operator sees a forecast that becomes more precise as the batch runs, giving earlier and more reliable warning when trending above target.

---

### Component 2.5 — The Confidence Scorer

**One job:** Attach a single, honest confidence score to every prediction that reflects all sources of uncertainty in a number the operator can act on.

A prediction of 91.4 Newton with 94 percent confidence is completely different information from a prediction of 91.4 Newton with 41 percent confidence. Without a confidence score, both look identical to the operator.

The score starts at a base of 0.95 — the typical accuracy on well-represented inputs — and applies reductions for each source of uncertainty. An input outside the training distribution reduces confidence by 35 percent. Detected model drift reduces it by 10 percent. A derived feature that could not be computed cleanly reduces it by 15 percent. The result is displayed with a colour indicator: green above 0.85, amber between 0.60 and 0.85, red below 0.60.

---

## 6. Layer 3 — The Decision Engine

Layer 3 converts model outputs into human decisions, cost information, and operator actions.

---

### Component 3.1 — The Cost Translator

**One job:** Convert every energy prediction from kilowatt-hours into Indian Rupees, compute the variance against the batch cost target, and project monthly cost at the current rate.

The rupee-per-kilowatt-hour rate is a single configurable value. When the electricity tariff changes, one number changes and all projections update automatically. For every batch, the cost translator produces the predicted energy cost, the gap against the cost target, an estimated CO₂ emission figure at 0.82 kilograms per kilowatt-hour, the carbon variance, and a monthly projection across the planned daily batch count. This projection is the ROI figure: if π0's recommendations reduce energy by 6 percent, the monthly saving is a specific, auditable rupee amount.

---

### Component 3.2 — The Recommendation Engine

**One job:** Convert SHAP attribution into a specific, physical, machine-level instruction that an operator can follow without additional interpretation.

A model-level recommendation says "reduce compression force." A machine-level recommendation says "reduce compression force on Compression Unit B — left panel RPM dial, from position 7 to position 6, after the current compression cycle completes in approximately six minutes, which will save an estimated 3.4 kilowatt-hours and reduce hardness from 109 Newton to approximately 95 Newton, which remains within the acceptable range."

The second version is actionable. The first is not. The Recommendation Engine produces the second version by combining the top SHAP contributor with the machine configuration map and the quality impact model.

---

### Component 3.3 — The Alert Engine

**One job:** Monitor anomaly scores and forecast deviations, fire alerts at the correct severity level, and ensure every alert contains a complete, structured record.

Two thresholds trigger alerts. A forecast exceeding the energy target by more than 15 percent triggers a WARNING. Exceeding by more than 25 percent, or an anomaly score above 0.60, triggers a CRITICAL. Every alert record contains: the alert identifier, batch identifier, precise timestamp, alert type, severity level, plain-English description of what is wrong, technical detail of the detection, root cause from SHAP attribution, the specific physical action recommended including machine name, the estimated energy saving if the action is taken, and the quality impact of the recommended adjustment.

---

### Component 3.4 — The Alert Acknowledgement System

**One job:** Track the full lifecycle of every alert from firing to resolution and store that lifecycle permanently.

An alert that was fired but not seen is as important to the audit record as an alert that was followed and resolved. Both tell a story about the batch. If a batch exceeded energy target and the alert at minute 45 was never acknowledged, that fact is part of the explanation of why the batch exceeded target. Without the acknowledgement system, that fact is invisible.

Every alert passes through six states: fired, delivered, seen, acknowledged, acted-upon, and resolved. If an alert does not reach the seen state within the configured timeout, the system escalates to the supervisor automatically. If the supervisor does not acknowledge within five minutes, the plant manager is notified. Each escalation creates a linked record without modifying the original.

---

### Component 3.5 — The Golden Batch Manager

**One job:** Maintain the authorised reference batch that defines optimal performance for each product formulation, version-control all changes, and make the golden targets available to the anomaly detector and dashboard comparisons.

The golden batch is not a random historical batch. It is explicitly designated by an authorised person based on evidence that it achieved the best combination of quality outcomes and energy efficiency. The designation creates a formal record: which batch it is based on, the designating person's identity and role, the timestamp, the stated reason for selection, and the complete set of target values it establishes.

When a new golden batch is designated, the previous version is retired with a timestamp and the identity of the person making the change. It is never deleted. The complete history of which reference was active at any point in time is permanently preserved and restorable.

---

## 7. Layer 4 — The API Bridge

### Component 4.1 — FastAPI

**One job:** Receive requests from the three dashboards, validate all inputs before they reach the models, route requests to the correct component, and return structured responses.

FastAPI is the post office of π0. The intelligence components speak Python. The dashboards speak JavaScript. FastAPI sits between them, receives requests from the browser, validates every field before it can proceed, passes validated data to the correct component, receives the response, and sends it back in a format the browser understands.

The API exposes eight endpoints. One receives batch parameters and returns predictions with SHAP and cost translation. One receives parameters plus live data and returns an updated forecast. One receives recent power readings and returns anomaly score and fault classification. One takes a batch identifier and returns the full SHAP explanation in plain English. One takes a batch identifier and returns the complete audit record — every prediction, alert, acknowledgement, actual outcome, and model version. One returns the current golden batch targets. One accepts actual QC measurements and triggers the feedback loop engine. One returns current model performance metrics.

---

## 8. Layer 5 — The Three Dashboards

π0 serves three different people with completely different information needs from the same underlying data.

---

### Dashboard A — The Operator View

The operator view shows three numbers and one colour. The three numbers are batch progress as a percentage, energy status against target, and predicted quality outcome summarised as good, caution, or risk. The colour is green, amber, or red.

When an alert fires, the entire view transforms into an action card showing in plain language what is wrong, step-by-step physical instructions specifying which machine and which control to adjust, the estimated saving if the action is taken, and three buttons: done, not possible, and help. The button press is the acknowledgement. The timestamp and operator identity are recorded from the active login session automatically.

---

### Dashboard B — The Technical View

Four panels. The pre-batch prediction panel contains the input form, five predictions with confidence intervals, a golden batch comparison delta, and the cost projection. The live energy monitor panel shows a real-time chart of actual versus predicted energy, a phase indicator, and the sliding window forecast with its confidence band. The power curve anomaly panel shows the live power and vibration readings, the anomaly score gauge, and fault type and recommendation when detected. The SHAP explanation panel shows the waterfall chart for each target, the plain-English attribution summary, and the specific recommendation derived from the top SHAP contributor.

---

### Dashboard C — The Management View

Four summary cards at the top: batches completed today, total energy against target, cost against target, and quality pass rate. Below, a seven-day energy trend chart against the target line. Below that, active alerts from the last 24 hours with resolution status. At the bottom, model health status with the current 30-day accuracy and the date of the last model update. Every number is a link that drills into the detail behind it.

---

## 9. The Feedback Loop

The feedback loop is what makes π0 a learning system rather than a static calculator. It consists of five stages forming a complete circle.

Stage one is prediction — five outcomes are predicted and stored with full provenance before every batch. Stage two is monitoring — the anomaly detector and forecaster produce real-time assessments and store every alert and acknowledgement during the batch. Stage three is outcome recording — after the batch completes, QC enters actual measured values for all five quality targets. Stage four is accuracy tracking — prediction error is computed for all five targets, the rolling 30-day accuracy is updated, and the management dashboard reflects the current model health. Stage five is the retraining decision — if rolling accuracy degrades beyond the threshold, the model is flagged for review and, when enough evidence accumulates, a retraining process is initiated, validated, staged, approved, and deployed. The loop returns to stage one with a more accurate model.

Without this loop, the model trained at launch is the model in production six months later regardless of how conditions have changed. With this loop, the model improves continuously as the system accumulates batch history.

---

## 10. The Audit Architecture

Every batch in a regulated pharmaceutical manufacturing environment must have a complete, unalterable record. π0's audit architecture is designed around this requirement.

The audit store maintains a complete record for every batch ever run. The record contains the full batch identity and timing, operator identity and login session, all input parameters and derived features with computation formulas, all five predictions with model version and confidence scores, the complete SHAP breakdown, all power curve readings for the entire batch duration, every anomaly score computed during the run, every alert fired with its complete content, every acknowledgement or non-acknowledgement with timestamps and operator identities, every action taken or declined with optional notes, actual QC measurements and prediction errors, and complete model lineage traceable to the training data.

The immutability rule is absolute. Once a batch is marked closed, no field in its audit record can be modified. Corrections are appended as new records linked to the original. This is append-only architecture — the foundation of every audit system that a regulatory body can rely upon.

---

## 11. The Alert and Acknowledgement System

An alert that is fired but never seen is invisible without a system to track it. The acknowledgement system makes the complete lifecycle of every alert explicit, permanent, and auditable.

Every alert passes through six states. Fired is when the alert is created. Delivered is confirmed by a WebSocket acknowledgement from the browser. Seen is inferred from the operator's active login session at the time of delivery. Acknowledged is confirmed by an explicit button press. Acted-upon is confirmed when the operator selects whether they followed, declined, or escalated the recommendation. Resolved is recorded when the batch phase associated with the alert completes.

If an alert does not move from fired to seen within ten minutes for a warning or three minutes for a critical, it escalates to the shift supervisor. If the supervisor does not acknowledge within five minutes, the plant manager receives a notification. Each escalation step creates a new linked record without modifying the original. The complete history of who was supposed to see it, who actually saw it, how long it took, and what was done is always queryable.

---

## 12. The Golden Batch System

The golden batch is the authorised reference standard for a product formulation. It is explicitly designated by an authorised person — typically the QC manager or senior process engineer — based on evidence that it achieved the best combination of quality and energy efficiency.

The designation creates a formal record: the batch identifier it is based on, the designating person's identity and role, the timestamp, the stated reason for selection, and the complete set of target values and parameter values it establishes. The anomaly detector uses the golden batch's power signature as its reconstruction reference. Every dashboard comparison is made against the golden batch targets.

Version control is maintained across all designations. When a new golden batch is designated, the previous version is marked retired with a timestamp. It is never deleted. If the new designation proves unsuitable, the previous version can be restored. The complete history of which reference was active at any point is permanently preserved.

---

## 13. Technology Stack

Every technology in π0 is described from first principles, compared to the alternatives considered, and justified with a specific reason for the choice made.

---

### Python

Python is the programming language in which all intelligence components are written.

Python is the universal language of machine learning and data science. Every library π0 relies on — XGBoost, SHAP, scikit-learn, PyTorch — was built primarily for Python and has its deepest support in Python. Using a different language would mean reimplementing these libraries from scratch or writing complex integration layers. Python also has the largest data science community, meaning the most tutorials, the most Stack Overflow answers, and the most examples for every problem encountered.

**Alternatives considered:** R is excellent for statistics but has no production deployment story. Julia is faster but the ecosystem is immature for manufacturing ML. Python wins on ecosystem size and deployment practicality.

---

### XGBoost

XGBoost is the machine learning algorithm powering the Multi-Target Predictor.

In plain language, XGBoost builds 200 decision trees where each tree learns from the mistakes of all previous trees. A decision tree is a sequence of if-then questions on your input features that eventually produces a numerical answer. The first tree makes rough predictions. Each subsequent tree focuses on the cases where the previous trees were wrong. After 200 trees, the combined answer is far more accurate than any single tree.

**Why not a neural network:** Neural networks are designed for problems where patterns are deeply hidden in complex unstructured data — images, audio, language. Your problem has seven inputs, sixty batches, and correlations above 0.99. The pattern is not hidden. A neural network on sixty rows would memorise the training examples rather than learn the pattern — a problem called overfitting — and perform worse than XGBoost on new batches.

**Why not linear regression:** Linear regression assumes the relationship between inputs and outputs is a straight line. The relationship between drying temperature and dissolution rate is not linear — very low and very high temperatures both produce worse dissolution rates than the optimal middle range. XGBoost makes no such assumption.

**Why not Random Forest:** Random Forest was the closest alternative and would have worked. XGBoost was chosen because it is consistently more accurate on small tabular datasets with high correlations, and it integrates natively with SHAP through a highly efficient exact algorithm called TreeExplainer.

---

### SHAP

SHAP is a Python library that explains what any machine learning model decided and why, by distributing the prediction across the input features.

The name stands for SHapley Additive exPlanations. The calculation uses a method from cooperative game theory for fairly dividing the outcome of a cooperative game among players who contributed different amounts. In π0, the players are the ten input features and the outcome being distributed is the deviation of the prediction from the average.

**Why not built-in feature importance:** XGBoost has a native feature importance metric, but it only tells you globally which features tend to matter across all predictions. SHAP tells you specifically how much each feature contributed to this specific prediction for this specific batch. That per-prediction granularity is what makes the Recommendation Engine work — and what makes the operator's explanation meaningful rather than generic.

**Why not LIME:** LIME is a similar explainability library. SHAP was chosen because SHAP produces stable, consistent explanations for the same input — running it twice on the same prediction produces identical values. LIME produces approximations that can vary between runs. For explanations stored as part of a permanent audit record, stability is a requirement.

---

### PyTorch with LSTM Autoencoder

PyTorch is a deep learning library. LSTM — Long Short-Term Memory — is a neural network architecture designed for sequential data where the order of readings matters.

Power curve data is sequential: the reading at second 900 is related to the reading at second 899. A standard neural network treats every input independently, losing all time relationship. An LSTM reads data the way you read a sentence — it maintains a memory of what came before and uses that memory to understand what it is currently reading. After processing an entire batch's power curve, the LSTM holds a compressed representation of that curve's character.

The autoencoder architecture extends this: the LSTM learns to compress a normal curve and then reconstruct it. When it encounters an abnormal curve, its reconstruction is inaccurate. That reconstruction error is the anomaly score.

**Why PyTorch not TensorFlow:** Both are excellent frameworks. PyTorch was chosen because its error messages are clearer during debugging, its intermediate values are more easily inspectable, and the manufacturing ML research community predominantly publishes PyTorch implementations.

---

### scikit-learn

scikit-learn is a Python library containing preprocessing tools, model evaluation utilities, and classic machine learning algorithms. π0 uses it for four specific purposes.

StandardScaler normalises the ten input features so no single parameter dominates the model's learning simply because its numerical range is larger. Without normalisation, compression force in kilonewtons and granulation time in minutes would have completely different scales, and the model would implicitly treat larger-numbered parameters as more important.

TimeSeriesSplit creates training and validation splits that respect time ordering. Randomly splitting 60 batches would allow batches from February to appear in training while batches from January appear in testing — leaking future information into past predictions. TimeSeriesSplit ensures test batches always come after all training batches chronologically.

MultiOutputRegressor wraps the five separate XGBoost models into a single interface, managing all five and returning all five outputs together when a prediction is requested.

Mean Absolute Percentage Error is the accuracy metric used by the Feedback Loop Engine to monitor model performance over time. It expresses prediction error as a percentage of the actual value, making it comparable across targets with different units and scales.

---

### FastAPI

FastAPI is a Python framework for building web APIs — the bridge between Python models and the JavaScript dashboard.

The intelligence components speak Python. The dashboards speak JavaScript. FastAPI creates a set of web addresses that the browser can send requests to. It receives the request, validates every field before it can proceed, passes the data to the correct component, receives the response, and returns it to the browser.

**Why not Flask:** Flask is the most popular Python web framework and would work. FastAPI was chosen for three reasons specific to π0. First, FastAPI validates all inputs using Pydantic — a text string accidentally sent where a number is expected is caught at the door with a clear error message, rather than crashing the model with a confusing traceback. Second, FastAPI is asynchronous — it handles a real-time sensor data stream and a prediction request simultaneously without either blocking the other. Flask is synchronous and would queue one behind the other. Third, FastAPI automatically generates interactive API documentation at the /docs endpoint with zero additional code — every endpoint, its required parameters, and its response format are documented and testable from any browser.

---

### React

React is a JavaScript library for building user interfaces that update automatically when their underlying data changes.

When a new power reading arrives every second during a live batch, the line chart must update, the anomaly gauge must update, and the sliding window forecast must refresh. In plain HTML with JavaScript, each of these updates would require manually locating the element, removing old data, inserting new data, and re-rendering — multiple lines of code per update across a dozen elements. React abstracts all of that. You define what the screen should look like for any given state of data, and React determines the minimum changes required to make the screen reflect new data. This is what makes the live charts smooth.

**Why not Vue:** Vue would work equally well technically. React was chosen because of ecosystem size. The Recharts charting library is built specifically for React and integrates without configuration. React also has the largest JavaScript community, meaning more relevant examples when problems arise.

---

### Recharts

Recharts is a charting library built specifically for React that provides the live power curve chart, sliding window forecast chart, SHAP waterfall chart, and management trend chart.

**Why not Chart.js:** Chart.js is a general JavaScript charting library not built for React — integrating it requires extra code to prevent React and Chart.js from both trying to manage the same parts of the page at the same time.

**Why not D3.js:** D3 is the most powerful data visualisation library in JavaScript but requires writing hundreds of lines of custom code for a single chart type. Recharts provides professional, configurable charts that update automatically when React data changes, in a format that requires minimal configuration.

---

### SQLite for Hackathon, PostgreSQL for Production

A database stores structured data in tables and allows it to be queried and retrieved reliably.

SQLite stores the entire database as a single file on disk. No server, no configuration. It works the moment the application starts. For a hackathon where the goal is a working demonstration of complete functionality, SQLite eliminates all database setup overhead.

PostgreSQL is a full database server for production deployment. It provides user authentication and role-based access control, so operators cannot modify records only QC should modify. It provides ACID compliance — if a write fails midway through, the entire write is automatically rolled back. It supports concurrent users writing simultaneously without conflicts. It supports the append-only table policies required by the audit architecture.

The transition from SQLite to PostgreSQL requires changing one configuration value. All application code communicates with the database through SQLAlchemy, a database abstraction library that speaks to both SQLite and PostgreSQL through the same interface. The application code is identical in both cases.

---

### Pydantic

Pydantic is a Python library for defining the structure and validation rules of data objects and automatically checking that incoming data conforms to those rules before it proceeds.

π0 defines formal schemas for every data type that crosses a boundary — batch parameters from the dashboard, prediction records written to the database, alert records from the alert engine. When data arrives at any boundary, Pydantic checks it against the schema. Any field that is missing, the wrong type, or outside its allowed range produces a specific, readable error message identifying exactly which field is wrong and why. Without Pydantic, bad data passes silently through multiple components before causing a failure whose error message points somewhere unhelpful.

---

### Docker Compose

Docker Compose runs the entire π0 application — backend, frontend, database — with a single command.

Without Docker, running π0 requires installing the correct version of Python, the correct version of Node, the database, all Python packages, all Node packages, and configuring both servers to communicate. One wrong library version breaks everything, and reproducing the error on a different machine may produce a different result because that machine has different background software. Docker packages the entire environment — every dependency, every configuration, the exact version of every library — into containers that run identically on any machine with Docker installed.

---

## 14. Hackathon vs Production Architecture

Every component in π0 has a hackathon version appropriate for demonstration and a production version appropriate for a real factory floor. They share the same interface — the upgrade is a component swap, not a rewrite.

| Component | Hackathon Version | Production Version |
|---|---|---|
| Data source | Two Excel files at startup | Process historian via OPC-UA real-time feed |
| Database | SQLite single-file | PostgreSQL with encryption and access controls |
| Audit records | Standard database writes | Append-only tables with cryptographic record hashing |
| Model serving | FastAPI in-process | Triton Inference Server, dedicated inference cluster |
| Anomaly detection | T001 golden signature plus synthetic curves | LSTM trained on thousands of real power curves per product |
| Alert delivery | Dashboard banner only | Dashboard plus SMS plus email plus supervisor escalation |
| Reporting | Real-time dashboard only | Scheduled PDF shift summaries, daily and weekly reports |
| Model retraining | Manual initiation by engineer | Automated pipeline with human approval gate before deployment |
| Compliance | SQLite records, no access control | 21 CFR Part 11 compliant with LIMS integration |
| Environments | Single environment | Development, staging, and production with approval gates |
| Multi-batch context | Each batch predicted independently | Inter-batch feature carryover from previous batch exit metrics |

---

## 15. Gap Audit

π0 names its limitations explicitly rather than hiding them. A judge or auditor with manufacturing experience will identify these gaps whether or not they are declared. Declaring them demonstrates architectural maturity and the ability to think critically about your own system.

---

**Gap A — Power Curve Coverage**

Only one real power curve exists in the provided data — batch T001. The LSTM anomaly detector is therefore trained on T001 as the golden reference plus synthetic curves generated for all other batches by scaling T001's phase signatures according to each batch's process parameters. The synthetic curves encode the physical relationships but they are not real sensor measurements. The production path is a process historian integration that captures real power curves for every batch automatically.

---

**Gap B — Energy as Derived Not Measured**

The batch production data file contains no measured energy column. The energy prediction target is computed from T001's phase-level consumption rates scaled by process parameters. It is a physically grounded estimate, not a metered measurement. The production path is smart meter integration per machine that writes directly to the prediction input pipeline.

---

**Gap C — Small Training Dataset**

The dataset contains 60 real batches. This is sufficient to produce a high-accuracy model because the correlations between inputs and outputs are extremely strong. However, it limits confidence in generalisation to parameter combinations that have never been seen. Domain-aware data augmentation expands the effective training set to 300 records within realistic bounds. Confidence scoring explicitly flags when a prediction uses inputs outside the historical range.

---

**Gap D — Single Machine**

The data provided covers one production line. π0 cannot model interactions between machines or predict how upstream deviations propagate to downstream outcomes. The production path is multi-machine cascade prediction using a graph-based model of the complete manufacturing line.

---

**Gap E — No Labelled Fault History**

The dataset contains no records of confirmed bearing failures, wet material batches, or calibration events. The fault classifier is trained on synthetic fault signatures constructed from domain knowledge. The production path is retrospective labelling of historical maintenance events combined with ongoing tagging of detected faults confirmed by maintenance records.

---

## 16. Future Scope

Eight features are defined for production development, each grounded in a specific identified gap and each specifying the technical path to implementation.

---

**FS-1 — Automated Model Retraining Pipeline**

Addresses the broken feedback loop. A scheduled pipeline detects accuracy degradation, trains a new model on all available batch history, validates it against a held-out set, stages it for five batches in parallel with the current model, and deploys upon human approval. The retraining process is fully auditable — every training run, every validation result, every deployment decision is stored permanently. Technical path: Apache Airflow for pipeline scheduling, MLflow for model registry and experiment tracking.

---

**FS-2 — Process Historian Integration**

Addresses the power curve coverage gap. A real-time connector to the plant's sensor data historian via OPC-UA protocol captures power curves for every batch automatically. Within six months, π0 holds hundreds of real curves available for genuine LSTM training on confirmed fault patterns. Technical path: Python OPC-UA library, TimescaleDB for time-series storage at one-second resolution.

---

**FS-3 — Shift Summary and Reporting Engine**

Addresses the absence of a reporting layer identified from the plant manager perspective. At every shift end, a structured PDF report is generated automatically: batches run, energy consumed, cost variance, alerts fired with resolution status, anomalies detected, and model accuracy. Emailed automatically to plant manager and operations head. Technical path: ReportLab for PDF generation, Celery for scheduled task execution, SMTP for email delivery.

---

**FS-4 — RAG-Based Batch Intelligence Chatbot**

Enables operators and engineers to query the batch history in plain English. An operator asks "Why did batch T033 achieve the best dissolution rate this year?" and receives a specific, evidence-based answer sourced from the prediction store, SHAP records, and process parameters — no SQL knowledge required. Technical path: ChromaDB for vector storage of batch records, LangChain RAG chain, Claude Haiku for grounded answer generation.

---

**FS-5 — Inter-Batch Feature Carryover**

Addresses the batch independence assumption. The prediction for each batch incorporates exit metrics from the previous batch — final Milling vibration, energy overrun, time since last maintenance event — improving accuracy for consecutive batches on the same machine. Technical path: new feature table in PostgreSQL storing batch exit metrics, modified feature engineering pipeline to pull previous batch features.

---

**FS-6 — 21 CFR Part 11 Compliance Layer**

Addresses the regulatory audit gap. Append-only records, cryptographic signatures on every batch record, electronic signature for batch closure, role-based access control, and audit log export formatted for regulatory submission. Technical path: PostgreSQL append-only table policies, SHA-256 record hashing per batch, LIMS integration for laboratory data management.

---

**FS-7 — Multi-Machine Cascade Prediction**

Extends the architecture to model the entire manufacturing line as a connected system where upstream deviations propagate to downstream predictions. When the granulator runs hotter than usual, the dryer automatically receives an updated prediction for the material it is about to receive. Technical path: Neo4j graph database for machine relationship and material flow modelling, graph neural network for cascade prediction.

---

**FS-8 — Mobile Supervisor Alert App**

Delivers critical alerts to the supervisor's phone as push notifications with acknowledgement capability from the phone. A Progressive Web App — installable from any browser without an app store — delivers three notification tiers: critical, warning, and information. Technical path: Service Workers for PWA installation, Web Push API, three-tier notification routing.

---

## 17. The One Question π0 Can Always Answer

Every architectural decision in this document exists to ensure that π0 can answer this question for any batch that has ever run through the system:

> *"For batch T047 — what did π0 predict before it started, which parameter drove each prediction, was any alert fired during the run, did the operator acknowledge it, what did they do, what were the actual QC results, how wrong was the prediction, and which version of the model made that prediction?"*

Every part of this question maps directly to a component:

| Part of the question | Component that answers it |
|---|---|
| What did π0 predict | Prediction Store |
| Which parameter drove each prediction | SHAP Explainer output stored in Prediction Store |
| Was any alert fired | Alert Engine records |
| Did the operator acknowledge it | Alert Acknowledgement System lifecycle record |
| What did they do | Alert acted-upon state with operator note |
| Actual QC results | Outcome Recorder fields in Prediction Store |
| How wrong was the prediction | Feedback Loop Engine error computation |
| Which model version | Model Registry linked by model_version field in Prediction Store |

Every row above is a direct database query. The complete answer is available in under two seconds. For any batch. For any point in time. Permanently.

That is what a foolproof architecture looks like.

---

*π0 — Pharmaceutical Intelligence Operations Platform*
*Built for pharmaceutical tablet manufacturing. Designed from five perspectives. Honest about its limits. Ready to learn.*
