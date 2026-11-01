# Fault Detection on the Tennessee Eastman Process (PCA T²/SPE + Random Forest)

Author: MOHAMMED ASADH, BTECH IN CPS, GOVERNMENT ENGINEERING COLLEGE THRISSUR.

A fault-monitoring pipeline for the Tennessee Eastman Process (TEP), a standard simulated chemical plant with 52 sensor/actuator variables and 20 fault types. It
1. raises an alarm when the plant leaves normal operation (PCA with Hotelling's T² and SPE statistics), and
2. names the fault with a Random Forest classifier, run only after an alarm is confirmed.

The PCA monitoring framework follows the multivariate statistical process monitoring approach reviewed in Venkatasubramanian, Rengaswamy, Yin and Kavuri (2003), *Computers & Chemical Engineering*, Part III. This project implements and evaluates that framework. It does not claim to invent it.

## Data
Rieth et al. (2017), "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation", Harvard Dataverse, DOI 10.7910/DVN/6C3JR1. The simulator is described in Downs and Vogel (1993).
Each run has 52 variables sampled every 3 minutes. Training runs have 500 readings with the fault starting at reading 20. Testing runs have 960 readings with the fault starting at reading 160. 500 runs per fault and 500 fault-free runs per file.
The data is not included in this repository. Download the four `.RData` files and place them in `DATA/`.

## Method
- **Normal model:** standardise the 52 variables using fault-free training data, then fit PCA. Keep 31 components, the smallest number explaining at least 90% of normal variance.
- **T²:** sum over the 31 components of (score² / component variance). Limit: chi-square distribution, 31 degrees of freedom, alpha = 0.01 (limit 52.19).
- **SPE:** squared reconstruction error from the 31 components. Limit: Box's scaled chi-square approximation using the variance of the 21 discarded components, alpha = 0.01 (limit 11.79).
- **Alarm rule:** a reading is flagged if T² or SPE exceeds its limit. An alarm is confirmed after 3 consecutive flagged readings.
- **Classifier:** 200-tree Random Forest on 20-reading windows (per-variable mean and standard deviation, 104 features). Trained on fault-free runs 1-500 and faulty runs 1-50 per fault from the training files, tested on runs 1-50 of the testing files. No run appears in both training and testing.

## Results

**Monitoring, 3-consecutive-alarm rule, 500 test runs per fault.** False alarms are measured on readings before the fault starts (readings 1-160). Missed % is the share of post-onset readings not flagged. Delay is readings from fault onset to the first confirmed alarm, averaged over runs where one occurred (1 reading = 3 minutes).

| Fault | Missed % | Delay | | Fault | Missed % | Delay |
|---|---|---|---|---|---|---|
| 1 | 0.5 | 5.0 | | 11 | 35.0 | 10.9 |
| 2 | 1.6 | 13.4 | | 12 | 2.0 | 10.7 |
| 3 | 99.9 | n/a | | 13 | 5.7 | 43.8 |
| 4 | 0.3 | 3.4 | | 14 | 0.3 | 3.3 |
| 5 | 77.5 | 3.2 | | 15 | 99.9 | n/a |
| 6 | 0.2 | 3.0 | | 16 | 84.0 | 44.5 |
| 7 | 0.2 | 3.0 | | 17 | 10.4 | 35.3 |
| 8 | 3.1 | 22.1 | | 18 | 6.6 | 53.0 |
| 9 | 99.9 | n/a | | 19 | 98.1 | 89.4 |
| 10 | 69.6 | 62.7 | | 20 | 53.3 | 46.4 |

The false-alarm rate before onset rounds to 0.0% for every fault with the 3-in-a-row rule (1.5% with a single-reading rule). The full tables are in `results/`.

**Faults 3, 9 and 15 are not detectable by this method.** On fault-free test runs, 28.8% of runs contain a 3-in-a-row alarm somewhere in readings 161-960, and faults 3, 9 and 15 reach 30.0%, 30.0% and 32.2%, the same rate as normal data. Their delay values are chance alarms and are not reported. Fault 19 shows a weak but real signal, with 1.9% of readings flagged against 0.07% in normal runs.

**Fault classification, 17 detectable faults (all except 3, 9, 15):** 92.5% window accuracy. Fault-free windows are classified as normal 99.4% of the time. Faults 3, 9 and 15 are classified as normal (0% recall), consistent with the monitoring result.
Some faults that PCA mostly misses (5, 11, 19) are classified at about 99%. The classifier is trained on labelled faults and PCA is not, so this is not a like-for-like comparison.

**Streaming demo** (`stream_demo.py`): one test run replayed one reading at a time. Fault 1 was confirmed 5 readings after onset and named correctly 17 readings later. Fault 4 was confirmed after 3 readings and named correctly. Fault 9 raised no alarm. Naming a fault takes about 17 further readings (about 51 minutes) because the classifier needs a full 20-reading window.

## Limitations
- Three of the 20 faults (3, 9, 15) are indistinguishable from normal operation for this method.
- Alarm limits assume independent, roughly Gaussian readings, which TEP does not strictly satisfy. The measured false-alarm rate (about 1% per statistic on fault-free test data) is in line with the target.
- A 3-reading confirmation rule trades about 2 extra readings of delay for near-zero false alarms.
- The classifier used 50 runs per fault, not 500, and is a single train/test split with one random seed. No hyperparameter tuning was done.
- No deep-learning comparison is included.

## Reproduce
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Convert the .RData files to parquet with `python convert_data.py`, then run in order: `check_data.py`, `look_faulty.py`, `check_faults.py`, `fit_pca.py`, `t2_spe.py`, `limits.py`, `evaluate.py`, `baseline.py`, `classify2.py`, `stream_demo.py`.