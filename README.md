Localized Wildlife & Agricultural Edge-AI Classifier

Project Overview

This project takes the challenge of monitoring invasive species and agricultural diseases in resource constrained environments. By bridging the gap between deep learning and edge hardware, this website enables offline, real-time classification for farmers and conservationists.

The project is built on an end-to-end pipeline:

Automated scraping of research grade imagery from iNaturalist to build localized datasets.

Implementation of INT8 dynamic quantization, reducing storage requirements and optimizing CPU latency.

Methodology & Tools

Training: PyTorch-based training with heavy data to improve model robustness to natural lighting and angles.

Deployment: Model export to ONNX format to ensure cross-platform compatibility on edge devices like Raspberry Pi.

Results & Optimization Analysis

[Benchmark Comparison](benchmark_comparison.png)

The quantization process successfully reduced the model's memory footprint. While the INT8 model offers significant space savings, the latency performance confirms the suitability of this lightweight model for edge hardware.
We used Generative AI Assistance when we were working on this project. We got help from AI tools like ChatGPT and GitHub Copilot. They helped us with a things:

* Automation Scripts: They assisted us with writing basic Python code to get good pictures, from the iNaturalist API.

* Code. Debugging: They wrote the versions of the scripts to export to ONNX and to make PyTorch use less space.

*. Structuring: They helped us make a plan. Organize the technical overview of the project analyze how well it works and set up the README. We used Generative AI Assistance to make these things better.
