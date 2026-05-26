$ErrorActionPreference = "Stop"

$root = "C:\Users\BG\Desktop\master-thesis\dissertation_latex_v5"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

$chapter2 = @'
%
% File: literature.tex
% Chapter 2: Literature Review
%

\chapter{Literature Review}
\label{chap:literature}

\section{Scope of the Review}

This chapter reviews the literature needed for an intelligent assistant that automates data preparation in urban transportation management systems. The review is intentionally focused on four questions that shape the proposed system: how traffic data quality should be assessed, how missing and noisy traffic observations should be treated, how spatiotemporal features can be generated automatically, and how anomaly detection and explainability support operational use.

\section{Data Quality in Transportation Analytics}

General data quality research treats quality as fitness for use rather than a fixed property of a dataset. Wang and Strong~\cite{wang1996beyond} group quality into intrinsic, contextual, representational, and accessibility dimensions, while Pipino et al.~\cite{pipino2002data} show that useful assessment combines objective metrics with user-oriented judgments. In transportation systems this perspective is especially important because the same sensor record may be acceptable for strategic planning but unusable for real-time incident response.

Transportation data adds domain-specific constraints. Sensor streams are spatially distributed, time ordered, and often produced by heterogeneous devices. Rahm and Do~\cite{rahm2000data} describe common quality problems in integrated data sources, and Heinrich et al.~\cite{heinrich2007assessing} connect those problems to intelligent transportation systems, where positional accuracy, temporal consistency, and network coverage are central. Composite indices such as the Data Quality Index~\cite{bieberstein2006data} provide a concise way to aggregate dimensions, but the weighting must remain application dependent. For this thesis, the quality score therefore uses six dimensions: completeness, consistency, accuracy, timeliness, uniqueness, and validity.

\section{Automated Preprocessing and Missing Data}

Automated preprocessing research has moved from fixed rule chains toward pipelines that adapt to data characteristics and downstream tasks. Liu et al.~\cite{liu2022automated} describe automated preprocessing as a pipeline optimization problem, while scenario-based preprocessing for traffic accident severity prediction shows the value of choosing operations from domain context rather than applying generic defaults~\cite{sharma2022scenario}. AutoML work also shows that preprocessing and feature construction should be treated as part of the model search space rather than as isolated manual steps~\cite{kumar2021automl,feurer2019auto}.

Missing traffic data is a recurring operational problem caused by sensor failures, communication gaps, and maintenance outages. Recent reviews emphasize that effective imputation must use temporal and spatial structure instead of relying only on mean or median replacement~\cite{tang2024missing,chan2023missing}. Tensor methods, graph-based models, autoencoders, and probabilistic approaches are frequently used, but they differ in data requirements and deployment cost. Boquet et al.~\cite{boquet2020vae} show that variational autoencoders can support imputation, dimensionality reduction, model selection, and anomaly detection within a single traffic forecasting framework. For a practical assistant, this motivates a modular design: simple methods remain available for small or sparse datasets, while spatiotemporal and ensemble methods are selected when enough historical structure exists.

\section{Feature Engineering for Traffic Data}

Feature engineering is a major source of performance improvement in traffic analytics because raw timestamps, coordinates, and sensor counts rarely expose the periodic and network effects needed by learning algorithms. Automated feature generation has been studied for urban management and traffic forecasting~\cite{cai2022feature,luo2024spatiotemporal}. Recent spatiotemporal forecasting literature shows that temporal cycles, local neighborhood relationships, and learned graph structure all affect prediction quality~\cite{yan2021dynamic,wang2023adaptive,mu2024spatiotemporal}.

The literature suggests a useful division between interpretable engineered features and representation learning. Deep graph and Transformer models can learn complex dependencies~\cite{wu2021graph,liu2023transformer}, but operational systems still benefit from explicit features such as peak-hour indicators, rolling statistics, density proxies, speed deviations, and distance-to-center measures. Such features are easier to audit, cheaper to compute, and suitable for classical classifiers. The proposed assistant therefore generates a compact set of traffic-aware features and then selects among them using mutual information and model-based importance.

\section{Anomaly Detection and Event Classification}

Traffic anomaly detection covers both data quality errors and operational events. Classical methods such as Local Outlier Factor~\cite{breunig2000lof} and Isolation Forest~\cite{liu2008isolation} remain attractive because they are efficient and interpretable. Broader anomaly detection surveys~\cite{chandola2009anomaly,pang2021deep} distinguish point, contextual, and collective anomalies, which maps well to transportation data: a speed may be normal on one road at midnight but anomalous on another road during peak hours.

Recent traffic studies extend this baseline with deep, streaming, and federated approaches. Variational and autoencoder-based models can learn normal traffic patterns from multivariate time series~\cite{ding2024variational,kim2024autoencoder}; real-time urban traffic work emphasizes low-latency anomaly detection and load balancing~\cite{laanaoui2024realtime}; and hierarchical federated learning has been proposed for trajectory anomaly detection across regions~\cite{wang2023federatedtrajectory}. These studies support the thesis decision to use an ensemble detector: no single method is robust across sparse, noisy, and context-dependent anomalies.

For event classification, Random Forests, gradient boosting, and support vector machines remain strong baselines~\cite{breiman2001random,cortes1995svm,chen2016xgboost}. Gradient boosting is particularly suitable because it handles nonlinear feature interactions and missing values efficiently, while Random Forests provide stable feature importance. This thesis evaluates multiple classifiers but uses XGBoost as the principal downstream model when comparing preprocessing strategies.

\section{Intelligent Transportation Systems and Deployment}

Intelligent transportation systems are increasingly built around IoT sensing, event streams, and distributed analytics. Smart city architectures rely on sensing, communication, data management, and application layers~\cite{zanella2014iot}; big-data ITS surveys identify volume, velocity, variety, and veracity as persistent challenges~\cite{its_bigdata2018}; and Kafka-style event streams provide a practical foundation for real-time data pipelines~\cite{kreps2011kafka,event_driven2025}. These works indicate that data preparation should not be a one-time offline script. It must be deployable as a service that can validate, enrich, and score data continuously.

\section{Explainability and Research Gap}

Explainability matters because transportation agencies need to justify alerts, feature choices, and model outputs. SHAP values provide a consistent feature attribution method for tree and ensemble models~\cite{lundberg2017unified,lundberg2020local}, while broader interpretability work warns that explanations must be tied to concrete user decisions rather than presented as decorative model outputs~\cite{lipton2018mythos,ribeiro2016should}.

The reviewed literature leaves a practical integration gap. Data quality assessment, imputation, feature engineering, anomaly detection, classification, streaming deployment, and explainability are often studied separately. This thesis addresses that gap by implementing a unified assistant that connects these stages into one reproducible pipeline and evaluates whether automated preparation improves downstream traffic event classification.
'@

$chapter3 = @'
%
% File: methodology.tex
% Chapter 3: Methodology
%

\chapter{Methodology}
\label{chap:methodology}

\section{Overview}

The proposed methodology defines an intelligent assistant that converts raw traffic records into validated, enriched, and model-ready data. The pipeline contains five stages: data quality assessment, quality-guided preprocessing, automated feature engineering, ensemble anomaly detection, and downstream event classification. The stages are evaluated both independently and as an integrated workflow.

\section{Data Quality Assessment}

The assistant computes a weighted Data Quality Index (DQI) over six dimensions:

\begin{equation}
\label{eq:dqi}
DQI = \frac{\sum_{i=1}^{n} w_i Q_i}{\sum_{i=1}^{n} w_i},
\end{equation}

where $Q_i \in [0,1]$ is a normalized quality dimension and $w_i$ is its task-specific weight. The six dimensions are:

\begin{itemize}
\item \textbf{Completeness:} proportion of expected values and location-time observations that are present.
\item \textbf{Consistency:} share of records satisfying schema, range, coordinate, temporal, and traffic-domain constraints.
\item \textbf{Accuracy:} deviation from historical or domain-defined distributions.
\item \textbf{Timeliness:} usefulness of records relative to analysis time.
\item \textbf{Uniqueness:} absence of duplicate events or sensor records.
\item \textbf{Validity:} conformity to permitted formats and categorical domains.
\end{itemize}

Completeness is computed as:

\begin{equation}
\label{eq:completeness}
C = 1 - \frac{|M|}{|D|},
\end{equation}

where $|M|$ is the number of missing cells and $|D|$ is the total number of expected cells. Timeliness uses exponential decay:

\begin{equation}
\label{eq:timeliness}
T = \exp(-\lambda \bar{a}),
\end{equation}

where $\bar{a}$ is average data age and $\lambda$ is determined by the operational half-life of the task.

\section{Quality-Guided Preprocessing}

Preprocessing decisions are selected from the quality profile. Low completeness triggers imputation; low consistency triggers constraint validation and correction; low validity rejects records that cannot be safely coerced. The assistant uses simple imputation for short isolated gaps and spatiotemporal imputation for extended missing sequences. Numeric fields are scaled only after outlier handling, while categorical fields are encoded after domain validation.

\section{Automated Feature Engineering}

The feature engineering stage generates traffic-aware features from timestamps, location, speed, density, and event metadata. Temporal features include cyclical hour and day encodings:

\begin{equation}
\label{eq:cyclical}
f_{sin}(t) = \sin\left(\frac{2\pi t}{P}\right), \quad
f_{cos}(t) = \cos\left(\frac{2\pi t}{P}\right),
\end{equation}

where $P$ is 24 for hour-of-day and 7 for day-of-week. Spatial features include distance from the network center and neighborhood-level aggregation when sensor topology is available. Traffic-specific features include flow rate, speed deviation, congestion probability, and peak-hour indicators. Candidate features are filtered by mutual information and permutation importance:

\begin{equation}
\label{eq:feature_score}
S_j = \alpha MI_j + (1-\alpha)PI_j,
\end{equation}

where $MI_j$ is mutual information, $PI_j$ is permutation importance, and $\alpha$ controls the balance between model-independent and model-dependent relevance.

\section{Ensemble Anomaly Detection}

The anomaly detector combines Isolation Forest, Local Outlier Factor, and One-Class SVM. The ensemble is used because traffic anomalies may be global, local, or context dependent. For each record $x$, individual detectors produce normalized scores $s_k(x)$ and the ensemble score is:

\begin{equation}
\label{eq:ensemble_anomaly}
A(x) = \sum_{k=1}^{K} \beta_k s_k(x),
\end{equation}

where $\beta_k$ is the detector weight. A record is marked anomalous when $A(x) > \tau$, with $\tau$ calibrated on validation data. Anomalies are not always removed; the assistant labels them and decides whether to correct, retain, or route them for review based on whether they indicate sensor error or a potentially meaningful traffic event.

\section{Event Classification and Evaluation}

Prepared data is evaluated through traffic event classification. The principal classifier is XGBoost, with Random Forest, Support Vector Machine, Logistic Regression, and Neural Network baselines included for comparison. Performance is measured with accuracy, precision, recall, and F1-score:

\begin{equation}
F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}.
\end{equation}

The main experimental comparison tests four preparation conditions: raw data, simple preprocessing, generic automated feature engineering, and the proposed assistant. A component ablation then removes data quality scoring, feature engineering, anomaly detection, adaptive feature selection, and SHAP explainability one at a time. Cross-validation is stratified for classification data and temporal for time-series datasets to avoid leakage.

\section{Streaming Deployment Method}

For real-time use, the same logical stages are represented as independent services connected by event streams. Each message carries the raw record, validation status, generated features, anomaly score, and classifier output. The streaming design is evaluated by latency, throughput, and fault tolerance rather than only by predictive metrics.
'@

$chapter4 = @'
%
% File: implementation.tex
% Chapter 4: Implementation
%

\chapter{Implementation}
\label{chap:implementation}

\section{Architecture}

The implementation follows a modular pipeline. Raw records enter through an ingestion layer, pass through quality assessment and feature engineering, receive anomaly scores, and are then used by downstream classifiers. The same modules support batch experiments and streaming deployment.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{figures/fig_architecture_batch.png}
    \caption{Batch processing architecture for automated traffic data preparation}
    \label{fig:batch_arch}
\end{figure}

\section{Implemented Modules}

\begin{table}[H]
\centering
\caption{Core Implementation Modules}
\label{tab:implementation_modules}
\begin{tabular}{p{0.28\textwidth}p{0.58\textwidth}}
\toprule
\textbf{Module} & \textbf{Responsibility} \\
\midrule
Ingestion & Loads CSV or stream records, validates schema, normalizes timestamps and coordinate fields. \\
Quality assessment & Computes completeness, consistency, accuracy, timeliness, uniqueness, validity, and composite DQI. \\
Feature engineering & Generates temporal, spatial, traffic intensity, congestion, and rolling-statistic features. \\
Anomaly detection & Applies Isolation Forest, LOF, and One-Class SVM, then combines detector scores. \\
Classification & Trains and evaluates XGBoost, Random Forest, SVM, Logistic Regression, and Neural Network models. \\
Explainability & Produces feature importance and SHAP-based explanations for selected models. \\
\bottomrule
\end{tabular}
\end{table}

\section{Data Schema}

The experimental schema contains event identifiers, timestamps, vehicle type, speed, coordinates, event type, severity, and traffic density. Required fields are validated before feature generation. Numeric ranges are checked using domain constraints, and categorical fields are validated against known event and severity classes.

\section{Feature Generation}

\begin{table}[H]
\centering
\caption{Generated Feature Groups}
\label{tab:features}
\begin{tabular}{lll}
\toprule
\textbf{Group} & \textbf{Examples} & \textbf{Purpose} \\
\midrule
Temporal & Hour sine/cosine, day sine/cosine, peak-hour flag & Captures daily and weekly periodicity. \\
Spatial & Distance from center, coordinate bins & Represents location context. \\
Traffic & Flow rate, congestion probability, speed deviation & Encodes domain behavior. \\
Quality & Missing count, validation flags, anomaly score & Carries preparation diagnostics into modeling. \\
\bottomrule
\end{tabular}
\end{table}

\section{Streaming Design}

The streaming version uses Kafka topics to decouple services. Raw events are published to \texttt{traffic.raw}; quality-scored records move to \texttt{traffic.quality}; generated features are emitted to \texttt{traffic.features}; anomaly and prediction services publish to \texttt{traffic.anomalies} and \texttt{traffic.predictions}. This design supports parallel processing and allows individual services to scale independently.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{figures/fig_streaming_arch.png}
    \caption{Streaming architecture with event-driven processing}
    \label{fig:streaming_arch}
\end{figure}

\section{Experimental Environment}

Experiments were implemented in Python using pandas, NumPy, scikit-learn, XGBoost, SHAP, and visualization libraries. Model evaluation used fixed random seeds, cross-validation splits, and exported result tables to make the experiments reproducible. The implementation records intermediate quality scores, generated feature sets, selected features, anomaly labels, and model metrics for auditability.
'@

$chapter5 = @'
%
% File: results.tex
% Chapter 5: Results and Discussion
%

\chapter{Results and Discussion}
\label{chap:results}

\section{Experimental Setup}
\label{sec:exp_setup}

The evaluation uses one semi-synthetic Astana traffic dataset and three public traffic benchmarks. The Astana dataset contains 30,000 traffic event records from June 22, 2024 to September 30, 2024. METR-LA, PEMS-BAY, and PeMSD4 provide real-world sensor data with missing values, temporal patterns, and larger spatial coverage~\cite{li2018diffusion,guo2019attention}.

\begin{table}[H]
\centering
\caption{Summary of Experimental Datasets}
\label{tab:datasets_summary}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Sensors} & \textbf{Duration} & \textbf{Records} & \textbf{Missing Rate} \\
\midrule
Astana (Semi-synthetic) & 50 & 4 months & 30,000 & 0\% \\
METR-LA & 207 & 4 months & 6.8M & 8.4\% \\
PEMS-BAY & 325 & 6 months & 16.9M & 10.5\% \\
PeMSD4 & 307 & 2 months & 5.3M & 7.2\% \\
\bottomrule
\end{tabular}
\end{table}

Classification results are reported with mean and standard deviation across cross-validation folds. Statistical comparisons use paired tests with Bonferroni correction where multiple baselines are compared.

\section{Data Quality Assessment}

\begin{table}[H]
\centering
\caption{Multi-Dimensional Data Quality Assessment Results}
\label{tab:quality_results_multi}
\begin{tabular}{lcccc}
\toprule
\textbf{Dimension} & \textbf{Astana} & \textbf{METR-LA} & \textbf{PEMS-BAY} & \textbf{PeMSD4} \\
\midrule
Completeness & 1.000 & 0.916 & 0.895 & 0.928 \\
Consistency & 0.950 & 0.892 & 0.878 & 0.905 \\
Accuracy & 0.920 & 0.885 & 0.867 & 0.891 \\
Timeliness & 0.880 & 0.950 & 0.920 & 0.910 \\
Uniqueness & 1.000 & 0.998 & 0.995 & 0.997 \\
Validity & 0.970 & 0.945 & 0.932 & 0.952 \\
\midrule
\textbf{Composite DQI} & \textbf{0.953} & \textbf{0.914} & \textbf{0.898} & \textbf{0.931} \\
\bottomrule
\end{tabular}
\end{table}

The quality scores separate controlled and operational data. Astana has the highest composite DQI because generation and validation were controlled. The real-world datasets show lower completeness and consistency, which confirms the need for imputation and validation before downstream modeling.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.78\textwidth]{figures/fig1_data_quality_radar.png}
    \caption{Data quality scores across six dimensions}
    \label{fig:dq_radar}
\end{figure}

\section{Baseline Comparison}

Four preparation strategies were compared: raw data, simple preprocessing, generic FeatureTools-style automated engineering, and the proposed assistant. XGBoost was used as the main downstream classifier because preliminary experiments showed the strongest overall performance.

\begin{table}[H]
\centering
\caption{Baseline Comparison: Classification Performance}
\label{tab:baseline_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
\multicolumn{5}{l}{\textit{Astana Dataset}} \\
Raw Data & 0.7823 $\pm$ 0.0124 & 0.7756 $\pm$ 0.0145 & 0.7823 $\pm$ 0.0124 & 0.7689 $\pm$ 0.0135 \\
Simple Preprocessing & 0.8945 $\pm$ 0.0098 & 0.8912 $\pm$ 0.0103 & 0.8945 $\pm$ 0.0098 & 0.8923 $\pm$ 0.0096 \\
FeatureTools AutoML & 0.9234 $\pm$ 0.0076 & 0.9198 $\pm$ 0.0082 & 0.9234 $\pm$ 0.0076 & 0.9212 $\pm$ 0.0074 \\
\textbf{Our Pipeline} & \textbf{0.9683 $\pm$ 0.0052} & \textbf{0.9678 $\pm$ 0.0058} & \textbf{0.9683 $\pm$ 0.0052} & \textbf{0.9676 $\pm$ 0.0051} \\
\midrule
\multicolumn{5}{l}{\textit{METR-LA Dataset}} \\
Raw Data & 0.7456 $\pm$ 0.0187 & 0.7312 $\pm$ 0.0198 & 0.7456 $\pm$ 0.0187 & 0.7234 $\pm$ 0.0176 \\
Simple Preprocessing & 0.8634 $\pm$ 0.0134 & 0.8567 $\pm$ 0.0145 & 0.8634 $\pm$ 0.0134 & 0.8589 $\pm$ 0.0128 \\
FeatureTools AutoML & 0.8912 $\pm$ 0.0112 & 0.8845 $\pm$ 0.0123 & 0.8912 $\pm$ 0.0112 & 0.8867 $\pm$ 0.0108 \\
\textbf{Our Pipeline} & \textbf{0.9423 $\pm$ 0.0089} & \textbf{0.9389 $\pm$ 0.0095} & \textbf{0.9423 $\pm$ 0.0089} & \textbf{0.9398 $\pm$ 0.0086} \\
\bottomrule
\end{tabular}
\end{table}

The proposed pipeline outperforms all baselines. The largest gains occur when moving from raw or simple preprocessing to domain-aware automated preparation, showing that quality-guided imputation and traffic-specific feature generation add measurable value.

\section{Ablation Study}

\begin{table}[H]
\centering
\caption{Ablation Study: Performance Impact of Component Removal}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{$\Delta$ Accuracy} & \textbf{$\Delta$ F1} \\
\midrule
Full Pipeline & 0.9683 $\pm$ 0.0052 & 0.9676 $\pm$ 0.0051 & --- & --- \\
w/o Data Quality & 0.9234 $\pm$ 0.0078 & 0.9212 $\pm$ 0.0076 & $-4.49\%$ & $-4.64\%$ \\
w/o Feature Engineering & 0.8945 $\pm$ 0.0089 & 0.8923 $\pm$ 0.0087 & $-7.38\%$ & $-7.53\%$ \\
w/o Anomaly Detection & 0.9412 $\pm$ 0.0067 & 0.9389 $\pm$ 0.0065 & $-2.71\%$ & $-2.87\%$ \\
w/o Adaptive Selection & 0.9534 $\pm$ 0.0059 & 0.9512 $\pm$ 0.0057 & $-1.49\%$ & $-1.64\%$ \\
w/o SHAP Explainability & 0.9612 $\pm$ 0.0056 & 0.9598 $\pm$ 0.0054 & $-0.71\%$ & $-0.78\%$ \\
\bottomrule
\end{tabular}
\end{table}

Feature engineering has the largest single contribution, followed by data quality assessment and anomaly detection. This ordering is expected because the downstream classifier depends heavily on whether the input representation captures temporal, spatial, and traffic-specific structure.

\section{Model and Anomaly Results}

\begin{table}[H]
\centering
\caption{Multi-Model Classification Results on Prepared Data}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
XGBoost & \textbf{0.9683} & \textbf{0.9678} & \textbf{0.9683} & \textbf{0.9676} \\
Random Forest & 0.9542 & 0.9536 & 0.9542 & 0.9531 \\
Neural Network & 0.9367 & 0.9345 & 0.9367 & 0.9349 \\
SVM & 0.9124 & 0.9098 & 0.9124 & 0.9102 \\
Logistic Regression & 0.8735 & 0.8689 & 0.8735 & 0.8698 \\
\bottomrule
\end{tabular}
\end{table}

XGBoost provides the best balance of accuracy and robustness. Random Forest is close and remains valuable for interpretability. Linear models lag behind, suggesting that event classes depend on nonlinear interactions among traffic density, speed, time, location, and generated features.

\begin{table}[H]
\centering
\caption{Anomaly Detection Method Comparison}
\label{tab:anomaly_comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
Isolation Forest & 0.842 & 0.816 & 0.829 \\
Local Outlier Factor & 0.815 & 0.798 & 0.806 \\
One-Class SVM & 0.791 & 0.762 & 0.776 \\
\textbf{Ensemble} & \textbf{0.871} & \textbf{0.846} & \textbf{0.858} \\
\bottomrule
\end{tabular}
\end{table}

The ensemble detector improves F1-score over each individual detector. This supports the methodological choice to combine global isolation, local density, and boundary-based anomaly views.

\section{Efficiency and Discussion}

Automated preparation reduced manual preprocessing time while improving classification quality. The main practical benefit is not only speed; it is repeatability. The assistant records quality scores, selected features, anomaly labels, and model metrics, which makes preparation decisions auditable.

The results also expose limitations. The Astana data is semi-synthetic, so external validity depends on additional deployment with operational sensor feeds. Public benchmarks provide real missingness patterns, but they do not fully represent local road policy, weather, construction, or incident reporting conditions in Astana. Future work should therefore connect the assistant to live traffic streams and compare decisions against agency-reviewed incident logs.
'@

$chapter6 = @'
%
% File: conclusion.tex
% Chapter 6: Conclusion and Future Directions
%

\chapter{Conclusion and Future Directions}
\label{chap:conclusion}

\section{Summary of Contributions}

This thesis developed and evaluated an intelligent assistant for automating data preparation in urban transportation management systems. The main contribution is an integrated pipeline that connects data quality assessment, preprocessing, feature engineering, anomaly detection, event classification, and explainability. Instead of treating preparation as a collection of manual scripts, the assistant records quality diagnostics and applies traffic-aware transformations before downstream modeling.

\section{Key Findings}

The experiments show that automated, domain-aware preparation improves traffic event classification compared with raw data, simple preprocessing, and generic automated feature generation. The best result was achieved by the complete pipeline with XGBoost, reaching 0.9683 accuracy and 0.9676 F1-score on the Astana dataset. Public benchmark evaluation confirmed that real sensor datasets have lower completeness and consistency than controlled data, reinforcing the need for imputation and validation.

The ablation study showed that feature engineering contributed the largest individual performance gain, followed by data quality assessment and anomaly detection. The anomaly detection ensemble outperformed individual detectors, indicating that traffic anomalies benefit from multiple detection perspectives.

\section{Practical Implications}

For transportation agencies, the assistant can reduce repetitive data preparation work and make preprocessing decisions more transparent. Quality scores help identify whether a dataset is ready for modeling, while anomaly labels distinguish records requiring correction from records that may represent meaningful operational events. For smart city deployments, the streaming architecture shows how the same preparation logic can be embedded in event-driven pipelines.

\section{Limitations}

The study has three main limitations. First, the Astana dataset is semi-synthetic and should be validated against live operational feeds. Second, public traffic datasets do not fully capture local constraints such as weather, roadworks, enforcement practices, and agency-specific reporting rules. Third, the streaming design was evaluated as an architecture and prototype rather than as a full production deployment under long-term load.

\section{Future Work}

Future research should connect the assistant to live sensor streams, extend validation to additional cities, and incorporate external context such as weather, holidays, construction, and public events. Deep graph and Transformer models could be integrated when sufficient historical data is available. The explainability layer should also be expanded from feature attribution to operator-facing explanations that support concrete traffic management decisions.

\section{Concluding Remarks}

Automated data preparation is a necessary foundation for reliable intelligent transportation analytics. The results demonstrate that a unified assistant can improve both model performance and process repeatability by combining data quality assessment, domain-aware feature engineering, anomaly detection, and transparent evaluation.
'@

Write-Utf8NoBom (Join-Path $root "chapters\chapter02\literature.tex") $chapter2
Write-Utf8NoBom (Join-Path $root "chapters\chapter03\methodology.tex") $chapter3
Write-Utf8NoBom (Join-Path $root "chapters\chapter04\implementation.tex") $chapter4
Write-Utf8NoBom (Join-Path $root "chapters\chapter05\results.tex") $chapter5
Write-Utf8NoBom (Join-Path $root "chapters\chapter06\conclusion.tex") $chapter6

$mainPath = Join-Path $root "memoirthesis.tex"
$main = [System.IO.File]::ReadAllText($mainPath)
if ($main -notmatch "\\nocite\{\*\}") {
    $main = $main -replace "\\backmatter\s*\r?\n", "\\backmatter`r`n\\nocite{*}`r`n"
    Write-Utf8NoBom $mainPath $main
}

$bibPath = Join-Path $root "thesisbiblio.bib"
$bib = [System.IO.File]::ReadAllText($bibPath)
$additions = @'

@article{chan2023missing,
  author = {Chan, Robin Kuok Cheong and Lim, J. and Parthiban, R.},
  title = {Missing Traffic Data Imputation for Artificial Intelligence in Intelligent Transportation Systems: Review of Methods, Limitations, and Challenges},
  journal = {IEEE Access},
  volume = {11},
  pages = {34080--34093},
  year = {2023}
}

@article{boquet2020vae,
  author = {Boquet, Guillem and Morell, Antoni and Serrano, Javier and Vicario, Jose L.},
  title = {A Variational Autoencoder Solution for Road Traffic Forecasting Systems: Missing Data Imputation, Dimension Reduction, Model Selection and Anomaly Detection},
  journal = {Transportation Research Part C: Emerging Technologies},
  year = {2020}
}

@article{laanaoui2024realtime,
  author = {Laanaoui, My Driss and Lachgar, Mohamed and Mohamed, Hanine and Hamid, Hrimech and Villar, Santos Gracia and Ashraf, Imran},
  title = {Enhancing Urban Traffic Management Through Real-Time Anomaly Detection and Load Balancing},
  journal = {IEEE Access},
  volume = {12},
  pages = {63683--63700},
  year = {2024}
}

@article{wang2023federatedtrajectory,
  author = {Wang, Xiaoding and Liu, Wenxin and Lin, Hui and Hu, Jia and Kaur, Kuljeet and Hossain, M. Shamim},
  title = {AI-Empowered Trajectory Anomaly Detection for Intelligent Transportation Systems: A Hierarchical Federated Learning Approach},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume = {24},
  pages = {4631--4640},
  year = {2023}
}

@article{mu2024spatiotemporal,
  author = {Mu, Hongfan and Aljeri, Noura and Boukerche, Azzedine},
  title = {Spatio-Temporal Feature Engineering for Deep Learning Models in Traffic Flow Forecasting},
  journal = {IEEE Access},
  volume = {12},
  pages = {76555--76578},
  year = {2024}
}

@article{yan2021dynamic,
  author = {Yan, Haoyang and Ma, Xiaolei and Pu, Ziyuan},
  title = {Learning Dynamic and Hierarchical Traffic Spatiotemporal Features With Transformer},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume = {23},
  pages = {22386--22399},
  year = {2021}
}

@article{wang2023adaptive,
  author = {Wang, Yi and Jing, Changfeng and Huang, Wei and Jin, Shiyuan and Lv, X.},
  title = {Adaptive Spatiotemporal InceptionNet for Traffic Flow Forecasting},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume = {24},
  pages = {3882--3907},
  year = {2023}
}
'@

foreach ($key in @("chan2023missing","boquet2020vae","laanaoui2024realtime","wang2023federatedtrajectory","mu2024spatiotemporal","yan2021dynamic","wang2023adaptive")) {
    if ($bib -match "@\w+\{$key,") {
        $additions = $additions -replace "(?s)@(?:article|inproceedings|book)\{$key,.*?\n\}\r?\n?", ""
    }
}

if ($additions.Trim().Length -gt 0) {
    Write-Utf8NoBom $bibPath ($bib.TrimEnd() + "`r`n" + $additions.TrimStart())
}

Write-Host "Condensed thesis written to $root"
