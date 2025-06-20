# Fake News Detection

This project focuses on detecting fake news using Natural Language Processing (NLP) techniques and Large Language Models (LLMs). It aims to identify and classify news articles as real or fake by analyzing their content, leveraging modern machine learning methodologies and pretrained language models.

## Table of Contents
- [Research Papers](#research-papers)
- [Data Collection](#data-collection)
  - [ISOT Fake News Dataset](#isot-fake-news-dataset)
  - [WELFake Dataset](#welfake-dataset)

## Research Papers
The following research papers have been selected to guide this project:

- [**Feature Computation Procedure for Fake News Detection: An LLM‑based Extraction Approach**](https://www.researchgate.net/publication/392127130_Feature_computation_procedure_for_fake_news_detection_An_LLM-based_extraction_approach)
- [**Evidence‑Backed Fact Checking Using RAG and Few‑Shot In‑Context Learning with LLMs**](https://arxiv.org/pdf/2408.12060)

## Data Collection

### ISOT Fake News Dataset
- https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
Two balanced CSV files of political and world‑news articles published primarily in 2016–2017.

| File      | Articles | Class | Key Columns                              |
|-----------|----------|-------|------------------------------------------|
| `True.csv`| 12,600+  | Real  | `title`, `text`, `type`, `date`          |
| `Fake.csv`| 12,600+  | Fake  | `title`, `text`, `type`, `date`          |

*Real* articles were scraped from Reuters; *Fake* articles were sourced from outlets flagged by PolitiFact and Wikipedia. Original punctuation and spelling were preserved to maintain authenticity.

### WELFake Dataset
- https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

A merged corpus drawn from four public news collections (Kaggle, McIntire, Reuters, BuzzFeed Political).

| Metric                     | Value                                     |
|----------------------------|-------------------------------------------|
| Total Articles             | 72,134                                    |
| Real Articles (`label = 1`) | 35,028                                    |
| Fake Articles (`label = 0`) | 37,106                                    |
| Columns                    | `serial_number`, `title`, `text`, `label` |

The aggregation reduces over‑fitting risk and provides a larger, domain‑diverse training set.
