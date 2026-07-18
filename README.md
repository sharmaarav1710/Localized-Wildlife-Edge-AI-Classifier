Localized Wildlife & Agricultural Edge-AI Classifier

Project Overview

This project addresses the challenge of monitoring invasive species and agricultural diseases in resource constrained environments. By bridging the gap between deep learning and edge hardware, this pipeline enables offline, real-time classification for farmers and conservationists.

The project is built on an end-to-end pipeline:

Data Pipeline: Automated scraping of research grade imagery from iNaturalist to build localized datasets.
Optimization: Implementation of INT8 dynamic quantization, reducing storage requirements and optimizing CPU latency.

Methodology & Tools
Training: PyTorch-based training with heavy data (rotation, jitter, flipping) to improve model robustness to natural lighting and angles.

Deployment: Model export to ONNX format to ensure cross-platform compatibility on edge devices like Raspberry Pi.

Results & Optimization Analysis
[Benchmark Comparison](benchmark_comparison.png)

The quantization process successfully reduced the model's memory footprint. While the INT8 model offers significant space savings, the latency performance confirms the suitability of this lightweight model for edge hardware.

Evaluation
Real world performance was assessed against an unseen holdout dataset, resulting in a baseline accuracy of 40%. This reflects the challenge of deploying models on chaotic, in-the-wild imagery compared to research-grade repository data.

Acknowledgments
This project was developed to explore hardware software co design for AI optimization in ecological contexts.