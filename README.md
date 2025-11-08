# Quarter 1 Project Checkpoint

## SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models

So far this quarter, we have been working with the code that goes along with this [paper](https://arxiv.org/abs/2403.02509) and testing it on different datasets. This repo contains the original code with some modifications that I made to enable easier testing. For example, I changed the model that is used from an OpenAI model to a Llama model, and made some other changes to different files.

Below is the original description of the algorithm that was written by the authors of the paper:

SPUQ is an LLM uncertainty calibration algorithm. It provides a confidence score for each query, for a given LLM.
Experiments show that this confidence score is correlated with the generation accuracy, and therefore provides a useful LLM response evaluation metric on-the-fly.

The details of the approach are documented in our [paper](https://arxiv.org/abs/2403.02509) published at EACL-2024 Conference.

The basic idea is to check whether an LLM provides a significantly different answer when we ask the same question in a slightly different way.
If it does, we assume the LLM is not confident in this case.
SPUQ perturbs the input (including the prompt and the temperature) to get multiple outputs, and then aggregate the outputs to obtain the final confidence score.
This allows SPUQ to address both epistemic (via perturbation) and aleatoric (via sampling) uncertainties, and it provides better calibration than some of the other existing methods.
