# Fake News Detection

This project focuses on detecting fake news using Natural Language Processing (NLP) techniques and Large Language Models (LLMs). It aims to identify and classify news articles as real or fake by analyzing their content, leveraging modern machine learning methodologies and pretrained language models.

## Table of Contents
- [Research Papers](#research-papers)
- [Data Collection](#data-collection)
- [Data Pre-Processing](#data-pre-processing)
- [Text Processing](#text-processing)

### Research Papers
The following research papers have been selected to guide this project wiht multiple approaches:
##### Approach 1:
- [**Feature Computation Procedure for Fake News Detection: An LLM‑based Extraction Approach**](https://www.researchgate.net/publication/392127130_Feature_computation_procedure_for_fake_news_detection_An_LLM-based_extraction_approach)
- [**Enhancing Fake News Detection with Word Embedding: A Machine Learning and Deep Learning Approach**](https://www.mdpi.com/2073-431X/13/9/239)
- [**WELFake: Word Embedding Over Linguistic Features for Fake News Detection**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9395133)
##### Approach 2:
- [**Evidence‑Backed Fact Checking Using RAG and Few‑Shot In‑Context Learning with LLMs**](https://arxiv.org/pdf/2408.12060)

# Approach 1 

### Data Collection

##### 1. ISOT Fake News Dataset
 - Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Two balanced CSV files of political and world‑news articles published primarily in 2016–2017.

| File      | Articles | Class | Key Columns                              |
|-----------|----------|-------|------------------------------------------|
| `True.csv`| 12,600+  | Real  | `title`, `text`, `type`, `date`          |
| `Fake.csv`| 12,600+  | Fake  | `title`, `text`, `type`, `date`          |

*Real* articles were scraped from Reuters; *Fake* articles were sourced from outlets flagged by PolitiFact and Wikipedia. Original punctuation and spelling were preserved to maintain authenticity.
 The cleaned text is then written back to the DataFrame, ready for feature extraction and model training.

##### 2. WELFake Dataset
 - Link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

A merged corpus drawn from four public news collections (Kaggle, McIntire, Reuters, BuzzFeed Political).

| Metric                     | Value                                     |
|----------------------------|-------------------------------------------|
| Total Articles             | 72,134                                    |
| Real Articles (`label = 1`) | 35,028                                    |
| Fake Articles (`label = 0`) | 37,106                                    |
| Columns                    | `serial_number`, `title`, `text`, `label` |

The aggregation reduces over‑fitting risk and provides a larger, domain‑diverse training set.

### Data Pre-Processing 

The preprocessing pipeline prepares raw news data into a clean and structured format suitable for training. It consists of the following steps:

1. **Loading libraries and datasets**  
   Required libraries are imported (e.g. Pandas, Scikit‑Learn, PyTorch, Seaborn, Matplotlib, WordCloud).  
   `Fake.csv` and `True.csv` (ISOT dataset) are loaded, labeled (`0 = fake`, `1 = real`), and concatenated.

2. **Integrating merged dataset**  
   A third dataset (`merge.csv`) is loaded, relabeled, cleaned (original `label` column renamed to `Label`), and appended to the main dataset.

3. **Handling missing values**  
   Missing values in the dataset are counted and any rows with nulls are handled or removed as appropriate.

4. **Dropping unnecessary columns**  
   Columns such as `subject` and `date` are removed to retain only the text and label fields.

5. **Removing duplicates**  
   Duplicate entries are identified and dropped to avoid redundancy.

6. **Shuffling data**  
   The dataset is shuffled with a fixed random seed (`42`) to ensure a reproducible split.

7. **Saving cleaned dataset**  
   The resulting DataFrame is saved to `data.csv` for downstream use.

8. **Exploratory Data Analysis (EDA)**  
   - A histogram of the label distribution is plotted.  
   - Token count distributions for `title` and `text` are visualized.  
   - Word clouds are generated separately for fake vs. real news in both titles and full text, highlighting frequently used words.

9. **Text cleaning**  
   Both `title` and `text` fields are cleaned using a `clean_text()` function that:
   - Converts text to lowercase  
   - Removes line breaks  
   - Strips digits and punctuation  
   - Collapses multiple spaces into one

   After the data is saved, the texts are further processed

 ### Text Processing

This pipeline prepares news text data (e.g., for fake news detection) by cleaning, processing, and analyzing both content and metadata.

1. **Contraction Handling**
   - Fixes broken contractions (e.g., `couldn t` → `couldn't`)
   - Expands standard contractions using custom regex patterns and the `contractions` library

2. **Text Cleaning Steps**
    Text is cleaned by:
     - Removing social media artifacts (handles, hashtags, URLs)
     - Stripping special characters and fixing formatting issues
     - Converting to lowercase and removing repeated content

3. **Advanced Token Processing**
    - Tokenizes text into words
    - Optionally removes stopwords
    - Applies stemming or lemmatization
    - Filters out punctuation and short tokens (≤ 2 characters)

4. **DataFrame Preprocessing**
    - Applies cleaning to both `text` and `title` columns
    - Adds cleaned versions and calculates text lengths
    - Filters out very short entries (less than 50 characters)

5. **Final Dataset Preparation**
    - Saves cleaned outputs to:
      - `final_data.csv` — simplified dataset for model input

6. **Exploratory Data Analysis**
    - Analyzes label distribution
    - Visualizes text/word length distributions
    - Highlights class imbalance and structural data issues

7. **Short Text Detection for Removal**
    - Identifies short entries :
      - `text` ≤ 80 characters
      - `title` ≤ 20 characters
    - Displays sample entries, counts, and visual summaries and removes them
