Localized Wildlife & Agricultural Edge-AI Classifier

Project Overview
This project addresses the problem of tracking and identifying the invasive species and the agricultural diseases in limited resources scenarios. Through connecting deep learning to edge devices, this site allows offline and real-time classification for farmers and conservationists.

The project utilizes a complete pipeline framework:

Automatic web scraping of scientific-grade images from iNaturalist for localized dataset creation.

Utilizes INT8 dynamic quantization that saves storage space and decreases CPU latency.

[Benchmark Comparison](benchmark_comparison.png)

The quantization process successfully reduced the model's memory footprint. While the INT8 model offers significant space savings, the latency performance confirms the suitability of this lightweight model for edge hardware.

We used Generative AI Assistance when we were working on this project. We got help from AI tools like ChatGPT and GitHub Copilot. They helped us with a things:

* Automation Scripts: They assisted us with writing basic Python code to get good pictures, from the iNaturalist API.

* Code. Debugging: They wrote the versions of the scripts to export to ONNX and to make PyTorch use less space.

* Structuring: They helped us make a plan. Organize the technical overview of the project analyze how well it works and set up the README. We used Generative AI Assistance to make these things better.

