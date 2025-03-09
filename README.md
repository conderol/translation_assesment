# LLM Translation Quality Assessment

This repository contains a Python application for evaluating the translation quality of various Language Models (LLMs) on product reviews. The system generates authentic-sounding product reviews with different sentiments in multiple languages, translates them using different models, and evaluates the translation quality using a more powerful LLM as an objective judge.

## Overview

Machine translation is a critical application for businesses with global reach. This project helps answer important questions about LLM translation capabilities:

- How do different LLM models compare in translation quality?
- Which models provide the best balance of quality, speed, and cost?
- How do models perform across different language pairs and sentiment types?
- Is using "LLM as a judge" an effective evaluation methodology?

## Features

- **Review Generation**: Creates realistic product reviews with positive, neutral, or negative sentiment in English, Tamil, and Chinese
- **Multi-Model Translation**: Translates content using three different models:
  - Claude 3.5 Haiku
  - Amazon Nova Micra
  - Amazon Nova Lite
- **Target Languages**: Translates to Russian, Hebrew, and German
- **Comprehensive Evaluation**: Uses Claude 3.7 Sonnet as a judge to evaluate translations on:
  - Correctness (accuracy of meaning, technical terms, product details)
  - Sentiment preservation
  - Word count ratio analysis
- **Performance Metrics**: Captures latency, token usage, and quality metrics for each translation

## Key Findings

Based on the evaluation results in the included CSV file:

### Claude 3.5 Haiku
- **Strengths**: Highest accuracy scores across most language pairs, excellent sentiment preservation
- **Weaknesses**: Highest latency (3-6 seconds on average), sometimes adds extra context
- **Best for**: Critical translations where accuracy is paramount, especially complex language pairs

### Nova Micra
- **Strengths**: Good balance of quality and speed (1-2 second latency), consistent performance
- **Weaknesses**: Occasional minor mistranslations, slightly lower accuracy than Haiku
- **Best for**: Most general-purpose translation needs with good quality requirements

### Nova Lite
- **Strengths**: Lowest latency (1-2 seconds), good performance on common language pairs
- **Weaknesses**: More inconsistent quality, occasional significant errors in complex language pairs
- **Best for**: High-volume use cases where speed is prioritized over perfect accuracy

## Language Pair Analysis
- **English → Target Languages**: All models performed well, with minimal quality differences
- **Tamil → Target Languages**: Larger performance gaps between models, with Haiku showing clear advantages
- **Chinese → Target Languages**: Mixed results, with Nova Micra sometimes outperforming Haiku

## LLM as a Judge

This project employs the concept of "LLM as a judge" - using a more powerful model (Claude 3.7 Sonnet) to evaluate the outputs of other models. This approach has several advantages:

- **Consistency**: Provides uniform evaluation criteria across all translations
- **Expertise**: Claude 3.7 demonstrates strong multilingual capabilities and understanding of translation quality
- **Efficiency**: Automates what would otherwise be a highly manual evaluation process
- **Nuanced Feedback**: Goes beyond simple metrics to provide qualitative feedback

The evaluation focuses on three key metrics:

- **Correctness Score (1-10)**: Accuracy of meaning, preservation of technical details
- **Sentiment Score (1-10)**: How well the emotional tone is maintained
- **Word Count Ratio**: Analysis of translation length appropriateness

## Usage

```python
# Set up AWS credentials before using the program
# Run the complete evaluation pipeline
run_me()
```
# Requirements
- Python 3.8+
- AWS SDK for Python (Boto3)
- AWS account with Bedrock access
- Access permissions for:
  - Claude 3.5 Sonnet (for review generation)
  - Claude 3.5 Haiku (for translation)
  - Claude 3.7 Sonnet (for evaluation)
  - Amazon Nova Micra and Nova Lite models

# Future Work
- Expand to additional language pairs
- Incorporate more translation models
- Add cost analysis per translation
- Implement confidence scores for evaluations
- Create visualization dashboard for results

# License
This project is licensed under the MIT License 
