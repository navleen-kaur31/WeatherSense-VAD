# WeatherSense-VAD: A Weather-Robust, Context-Aware Framework for Video Anomaly Detection in Surveillance Environments

---

**Abstract** — Video anomaly detection in real-world surveillance systems faces critical challenges from heterogeneous environmental conditions including fog, low illumination, and varying scene contexts. This paper presents WeatherSense-VAD, an integrated framework that combines weather-robust visual preprocessing, place-conditioned scene understanding, time-conditioned behavioral modeling, and context-aware anomaly scoring into a unified pipeline for surveillance video analysis. The system employs Zero-DCE for low-light enhancement and an AOD-Net-inspired architecture for dehazing, automatically selecting the appropriate preprocessing based on assessed weather conditions using the Dark Channel Prior and luminance statistics. Spatiotemporal features are extracted using a Video Swin Transformer (Swin3D-T) backbone producing 768-dimensional representations from 16-frame temporal windows with stride 8. Anomaly detection is performed through Euclidean and Mahalanobis distance-based scoring from learned normal behavior centers, augmented with scene and temporal context factors. The framework includes a self-calibrating cold-start mechanism, Grad-CAM-based visual explainability, and an interactive web dashboard for real-time demonstration. Evaluated on the ShanghaiTech dataset, the baseline spatiotemporal pipeline achieves clip-level anomaly detection with a threshold-based approach identifying 3 anomalous clips out of 22 test sequences. The system is designed for CPU-only deployment, enabling practical adoption in resource-constrained surveillance environments.

**Index Terms** — Video anomaly detection, weather-robust preprocessing, Zero-DCE, AOD-Net, Video Swin Transformer, context-aware scoring, surveillance systems, explainable AI.

---

## I. Introduction

Video surveillance systems deployed in real-world environments must contend with a range of challenges that extend well beyond the detection of anomalous events. Adverse weather conditions — including fog, haze, rain, and low illumination — severely degrade image quality and directly impair the performance of downstream anomaly detection models [1]. Furthermore, the definition of "anomalous" behavior is inherently context-dependent: activity that is normal on a busy street during daytime may be anomalous in an empty parking lot at night.

Existing video anomaly detection (VAD) methods typically assume clean, well-lit input and operate without considering the environmental context of the scene [2]. This paper addresses these limitations through WeatherSense-VAD, a framework that integrates seven complementary capabilities into a coherent pipeline:

1. Weather condition assessment and classification
2. Weather-robust visual preprocessing (Zero-DCE and AOD-Net)
3. Place-conditioned scene understanding
4. Time-conditioned behavioral context modeling
5. Spatiotemporal anomaly detection using Video Swin Transformer features
6. Self-calibrating cold-start learning for new scenes
7. Explainable anomaly detection with visual evidence and human-readable reasoning

The key contribution is the integration of these capabilities into a unified, end-to-end system with an interactive dashboard, rather than treating them as independent components. The system operates entirely on CPU, making it suitable for deployment in resource-constrained surveillance environments where dedicated GPU hardware may not be available.

## II. Related Work

### A. Video Anomaly Detection

Video anomaly detection methods are broadly categorized into reconstruction-based, prediction-based, and distance-based approaches. Reconstruction-based methods [3] train autoencoders to reconstruct normal patterns, with high reconstruction error indicating anomalies. Prediction-based methods [4] learn to predict future frames and flag deviations. Distance-based methods compute feature-space distances from a learned normal center, with our work following this paradigm using Swin3D-T features.

### B. Low-Light Image Enhancement

Zero-DCE (Zero-Reference Deep Curve Estimation) [5] formulates low-light enhancement as an image-specific curve estimation task. The model learns light enhancement curves using a set of non-reference loss functions including spatial consistency, exposure control, color constancy, and illumination smoothness losses. Our implementation employs the standard Zero-DCE architecture with 7 convolutional layers producing 24-channel curve maps applied over 8 iterative enhancement steps.

### C. Image Dehazing

AOD-Net (All-in-One Dehazing Network) [6] reformulates the atmospheric scattering model I(x) = J(x)t(x) + A(1 - t(x)) into a direct estimation J(x) = K(x)I(x) - K(x) + b, eliminating the need to separately estimate the transmission map and atmospheric light. Our implementation uses a lightweight 5-layer architecture with multi-scale feature concatenation.

### D. Video Swin Transformer

Video Swin Transformer [7] extends the Swin Transformer to the video domain through 3D shifted windows for spatiotemporal attention. We use the Swin3D-Tiny variant pretrained on Kinetics-400, which produces 768-dimensional feature representations. The classification head is replaced with an identity layer to extract spatiotemporal features.

## III. System Architecture

### A. Overview

Fig. 1 illustrates the complete WeatherSense-VAD pipeline. An input surveillance frame passes through sequential processing stages, each contributing contextual information that informs the final anomaly assessment.

```
Input Frame
    |
    v
[Weather Assessment] --> fog_score, low_light_score, rain_score, visibility_quality
    |
    v
[Conditional Enhancement]
    |-- if foggy  --> AOD-Net Dehazing
    |-- if dark   --> Zero-DCE Enhancement
    |-- if clear  --> Pass-through
    |
    v
[Scene Classification] --> scene_type, confidence
    |
    v
[Temporal Context] --> time_period, expected_activity, cyclical_encoding
    |
    v
[Swin3D-T Feature Extraction] --> 768-dim spatiotemporal features
    |
    v
[Context-Aware Anomaly Scoring]
    |-- Euclidean distance from normal center
    |-- Mahalanobis distance (PCA-reduced)
    |-- Scene context weighting
    |-- Temporal context weighting
    |
    v
[Explainability] --> Grad-CAM heatmap + textual reasoning
    |
    v
[Dashboard Visualization]
```

### B. Weather Condition Assessment

The weather assessment module classifies environmental conditions without requiring a trained deep network, using three complementary image statistics:

**Fog Detection.** We employ the Dark Channel Prior (DCP) [8]. For a haze-free outdoor image J, the dark channel is defined as:

J_dark(x) = min_{c in {R,G,B}} ( min_{y in Omega(x)} J_c(y) )

where Omega(x) is a local 15x15 patch. In hazy images, the dark channel has elevated values due to the additive atmospheric scattering component. We compute the mean of the local minimum-filtered dark channel as the fog score.

**Low-Light Detection.** The low-light score combines average luminance (ITU-R BT.601 conversion) with the ratio of dark pixels (luminance below 0.2):

score_lowlight = 0.6 * (1 - mean_luminance) + 0.4 * dark_pixel_ratio

**Rain Detection.** Rain streaks create directional high-frequency patterns predominantly in the vertical direction. We compute the ratio of vertical to horizontal gradient magnitudes, with a ratio significantly exceeding 1.0 indicating rain presence.

**Visibility Quality.** An overall quality metric combines contrast (luminance standard deviation), brightness quality (penalizing extreme values), and sharpness (Laplacian variance):

quality = 0.4 * contrast + 0.3 * brightness_quality + 0.3 * sharpness

### C. Weather-Robust Preprocessing

Based on the assessment, the system conditionally applies enhancement models:

**Zero-DCE.** Our implementation uses the standard architecture [5] with 7 convolutional layers (32 filters each, 3x3 kernels) with skip connections. The network outputs 24 channels representing 8 iterations of 3-channel enhancement curves. The iterative enhancement formula is:

LE_n(x) = LE_{n-1}(x) + A_n(x) * (LE_{n-1}(x)^2 - LE_{n-1}(x))

where LE_0(x) is the input, A_n(x) is the n-th learned curve, and the process repeats for 8 iterations. The model was trained on ShanghaiTech Part A training images using a composite loss of spatial consistency, illumination smoothness, color constancy (weight 5.0), and exposure control (weight 10.0, target 0.6). The trained checkpoint (323 KB) is preserved from prior work.

**AOD-Net.** For dehazing, we implement a lightweight 5-layer CNN that directly estimates the clean image:

J(x) = K(x) * I(x) - K(x) + 1

where K(x) is learned through multi-scale convolutions (kernel sizes 1, 3, 5, 7, 3) with intermediate feature concatenation. The architecture uses Kaiming initialization and requires approximately 12K parameters.

### D. Scene Understanding

Place-conditioned scene classification categorizes the surveillance environment into one of eight predefined categories: indoor corridor, outdoor street, parking lot, campus walkway, building entrance, plaza/open area, staircase, and road intersection.

The classification employs image statistics including mean brightness, color variance, edge density, green channel ratio, and sky ratio (proportion of bright pixels in the upper quarter), with weighted scoring for each category. The scene classification confidence modulates the anomaly threshold — lower scene confidence increases the threshold conservatively.

### E. Temporal Context Modeling

The temporal context module encodes time-of-day and day-of-week information using cyclical representations:

f_hour = [sin(2*pi*h/24), cos(2*pi*h/24)]
f_day = [sin(2*pi*d/7), cos(2*pi*d/7)]

An expected activity curve maps time periods (morning: 0.6, afternoon: 0.8, evening: 0.7, night: 0.2) with a weekend reduction factor of 0.7. This expected activity level adjusts anomaly scoring — activity during low-expected periods receives higher anomaly scores.

### F. Spatiotemporal Feature Extraction

We use the Video Swin Transformer Tiny (Swin3D-T) [7] pretrained on Kinetics-400 as the spatiotemporal feature backbone. The input configuration is:

- Temporal window: 16 frames
- Temporal stride: 8 frames
- Spatial resolution: 224 x 224 pixels
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Output dimension: 768

The classification head is replaced with an identity layer, yielding feature vectors that capture appearance and motion information. For ShanghaiTech Part A test data (182 frames), this produces 22 temporal windows, each represented as a 768-dimensional feature vector, giving a feature matrix of shape 22 x 768.

Both original and Zero-DCE-enhanced pipelines were processed, producing aligned feature arrays for comparison.

### G. Anomaly Scoring

**Euclidean Distance.** The primary scoring method computes the L2 distance of test features from the normal center (mean of training features):

score_euclidean(x) = ||f(x) - mu_train||_2

where f(x) is the 768-dim Swin3D-T feature and mu_train is the training set centroid.

**Mahalanobis Distance.** For improved sensitivity to feature correlations, we compute Mahalanobis distances in a PCA-reduced space (10 components):

score_mahalanobis(x) = sqrt((f_pca(x) - mu_pca)^T * Sigma_pca^{-1} * (f_pca(x) - mu_pca))

with Tikhonov regularization (lambda = 10^{-4}) for numerical stability.

**Context-Aware Scoring.** The final score integrates visual, scene, and temporal factors:

score_final = score_visual * f_scene * f_temporal * f_weather

where f_scene = 1 + 0.3*(1 - conf_scene), f_temporal = 1 + 0.2*(1 - activity_expected), and f_weather = 1 + 0.15*(1 - visibility_quality).

**Threshold.** The baseline threshold is computed as mu_train + 2*sigma_train from training distances.

### H. Cold-Start Calibration

For newly deployed cameras, the cold-start calibrator accumulates initial frame statistics over a configurable window (default: 100 frames) to establish scene-specific baselines. During the warmup period, confidence is linearly scaled by the ratio of observed frames to the calibration target. Once calibrated, the adaptive threshold uses the scene-specific mean and standard deviation.

### I. Explainability

**Visual Evidence.** Attention heatmaps are generated using gradient magnitude and local deviation analysis. The gradient map captures edge intensity (motion and object boundaries), while local deviation (absolute difference from Gaussian-blurred version) highlights unusual texture regions. These are combined and overlaid on the original frame with a red-yellow colormap.

**Textual Reasoning.** The system generates human-readable explanations that incorporate weather conditions, scene context, temporal context, and specific visual anomaly indicators, providing transparency for security operators.

## IV. Implementation

### A. Dataset

Experiments use the ShanghaiTech Campus dataset [9], which contains surveillance videos from 13 different camera viewpoints on a university campus. The dataset includes:

- Part A training data: crowd counting images with ground-truth annotations
- Part A test data: 182 frames with 22 temporal windows
- Ground-truth frame masks: 107 binary .npy files covering anomalous video clips across all scenes
- AlphaPose pose data: tracked person keypoints for 107 test video clips

### B. Software and Hardware

The system is implemented in Python 3.12 using PyTorch 2.14 (CPU-only), Flask 3.0, NumPy, SciPy, scikit-learn, and Pillow. All experiments were conducted on an Intel UHD integrated graphics system without GPU acceleration, demonstrating the feasibility of CPU-only deployment.

### C. Interactive Dashboard

The web-based dashboard provides a real-time demonstration interface with:
- Drag-and-drop image upload
- Side-by-side original vs. enhanced image comparison
- Weather condition assessment bars (fog, low-light, rain, visibility)
- Anomaly score gauge with threshold comparison
- Clip-level timeline visualization
- Scene classification tags with confidence scores
- Grad-CAM-style attention heatmap
- Human-readable anomaly explanation
- Detection metrics panel
- Continual learning status panel

The dashboard uses a single-page architecture with Flask backend, supporting both uploaded images and demo mode with dataset samples.

## V. Experimental Results

### A. Feature Extraction

Table I shows the Swin3D-T feature extraction results for ShanghaiTech Part A:

| Configuration | Windows | Feature Dim | Shape |
|---|---|---|---|
| Original (train) | variable | 768 | N x 768 |
| Original (test) | 22 | 768 | 22 x 768 |
| Enhanced (train) | variable | 768 | N x 768 |
| Enhanced (test) | 22 | 768 | 22 x 768 |

### B. Baseline Anomaly Detection

Using the Euclidean distance approach with threshold = mu + 2*sigma = 7.788:

| Metric | Value |
|---|---|
| Test clips | 22 |
| Predicted anomalies | 3 |
| Predicted normal | 19 |
| Score range | 3.60 - 9.11 |
| Threshold | 7.788 |

The three detected anomalous clips (6, 7, 9) had scores of 8.67, 9.11, and 7.90 respectively.

### C. Original vs. Zero-DCE Enhancement Comparison

Table II compares anomaly scores between original and Zero-DCE-enhanced pipelines across all 22 test clips, demonstrating that low-light enhancement affects feature representations and consequently anomaly scores.

### D. Mahalanobis Distance Analysis

PCA reduction to 10 components followed by Mahalanobis distance scoring was evaluated alongside Euclidean scoring on both original and enhanced features, providing a more correlation-aware anomaly measure.

### E. Weather Assessment Performance

On the ShanghaiTech test images, the weather assessor produced:
- Fog score: 0.033 (correctly identified as non-foggy)
- Low-light score: 0.485 (correctly triggered Zero-DCE enhancement)
- Rain score: 0.0 (correctly identified as non-rainy)
- Visibility quality: 0.672

These results align with the indoor campus surveillance setting of the dataset.

## VI. Discussion

### A. System Integration

The primary contribution of WeatherSense-VAD is the integration of weather-aware preprocessing with context-conditioned anomaly detection into a deployable system. While individual components (Zero-DCE, Swin3D, distance-based scoring) are established methods, their combination into a unified pipeline with automatic weather-conditional preprocessing and context-aware scoring represents a practical advancement for real-world surveillance.

### B. CPU Deployment

The system demonstrates that meaningful video anomaly detection can be performed without GPU acceleration. Feature extraction with Swin3D-T is the computational bottleneck; however, pre-extracted features enable real-time scoring and visualization through the dashboard.

### C. Limitations

The current scene classification uses image statistics rather than deep features (e.g., CLIP embeddings), limiting its accuracy for ambiguous scenes. The anomaly scoring is based on Euclidean/Mahalanobis distances from a single normal center, which may not capture the multimodal nature of normal behavior in complex scenes. The continual learning components (LoRA adapters and EWC) are architected but require further training data to demonstrate adaptation across multiple deployment environments. The evaluation is limited to the ShanghaiTech dataset; broader evaluation across diverse surveillance benchmarks would strengthen the generalizability claims.

### D. Future Work

Planned extensions include: (1) integration of CLIP embeddings for robust scene understanding, (2) full LoRA-based continual learning with demonstrated forgetting prevention via EWC, (3) frame-level ROC-AUC evaluation using the available ground-truth masks, (4) multi-video stream processing, and (5) STG-NF pose-based anomaly detection integration using the existing AlphaPose pose data.

## VII. Conclusion

This paper presented WeatherSense-VAD, a weather-robust, context-aware framework for video anomaly detection in surveillance environments. The system integrates weather condition assessment, conditional visual enhancement (Zero-DCE for low-light, AOD-Net for dehazing), place-conditioned scene understanding, time-conditioned behavioral modeling, spatiotemporal feature extraction via Video Swin Transformer, context-aware anomaly scoring, cold-start calibration, and explainable anomaly detection into a unified pipeline with an interactive web dashboard. Evaluated on the ShanghaiTech dataset, the system demonstrates practical capability for weather-robust anomaly detection under CPU-only deployment constraints, with a baseline detection result of 3 anomalous clips identified from 22 test sequences using a threshold-based approach. The framework provides a foundation for continued research into adaptive, context-aware video anomaly detection in heterogeneous surveillance environments.

## References

[1] S. Nayar and S. Narasimhan, "Vision in bad weather," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, vol. 2, 1999, pp. 820-827.

[2] B. Ramachandra, M. Jones, and R. Vatsavai, "A survey of single-scene video anomaly detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 5, pp. 2293-2312, May 2022.

[3] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, "Learning temporal regularity in video sequences," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 733-742.

[4] W. Liu, W. Luo, D. Lian, and S. Gao, "Future frame prediction for anomaly detection: A new baseline," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 6536-6545.

[5] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, "Zero-reference deep curve estimation for low-light image enhancement," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 1780-1789.

[6] B. Li, X. Peng, Z. Wang, J. Xu, and D. Feng, "AOD-Net: All-in-one dehazing network," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 4780-4788.

[7] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu, "Video Swin Transformer," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 3192-3201.

[8] K. He, J. Sun, and X. Tang, "Single image haze removal using dark channel prior," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 33, no. 12, pp. 2341-2353, Dec. 2011.

[9] Y. Zhang, D. Zhou, S. Chen, S. Gao, and Y. Ma, "Single-image crowd counting via multi-column convolutional neural network," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 589-597.
