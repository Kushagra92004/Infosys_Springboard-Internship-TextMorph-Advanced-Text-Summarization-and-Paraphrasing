
# TextMorph: Advanced Text Summarization and Paraphrasing

TextMorph is a comprehensive text summarization system developed as part of the Infosys Springboard Virtual Internship under the Generative AI module. The project's main goal is to build a system that can take long pieces of text and generate short, meaningful summaries using different AI models.

The project was implemented in **Google Colab** using **Hugging Face Transformers** and other NLP libraries. An interactive UI was created using `ipywidgets` to make it easier to test, compare, and visualize results from multiple models in real-time.

---

## 🎯 Aim

The aim of this project is to compare and analyze the performance of multiple AI models for **text summarization** and understand how different models handle meaning, fluency, and readability.

---

## ✨ Key Features

* **Diverse Techniques:** Explores and implements both **Abstractive** and **Extractive** summarization methods.
* **Multiple Models:** Integrates several pre-trained transformer models from the Hugging Face ecosystem.
* **Interactive UI:** Features a user-friendly interface designed in Google Colab for real-time summarization and comparison.
* **Comprehensive Evaluation:** Assesses model performance using a variety of metrics including ROUGE, Semantic Similarity, Readability, and Processing Time.
* **Robust Testing:** Compares model results on **10 different domains** of text to ensure a fair and diverse evaluation.

---

## 🤖 Models Implemented

The following models were integrated and compared in this project:

| Model | Type | Developed By | Description |
| :--- | :--- | :--- | :--- |
| **TinyLlama-1.1B-Chat** | Abstractive | TinyLlama Community | A lightweight version of LLaMA fine-tuned for chat and summarization tasks. |
| **Phi-2** | Abstractive | Microsoft | Compact 2.7B model trained on high-quality reasoning and educational datasets. |
| **BART-Large-CNN** | Abstractive | Meta (Facebook) | Transformer model fine-tuned on the CNN/DailyMail dataset for news summarization. |
| **Gemma-2B-IT** | Abstractive | Google DeepMind | Instruction-tuned model built for summarization and text generation. |
| **TextRank** | Extractive | NLTK / NetworkX | A classic algorithm that extracts key sentences based on graph ranking. |

---

## ⚙️ Tech Stack & Libraries

* **Environment:** Google Colab (GPU Runtime)
* **Language:** Python 3.12
* **Core Libraries:**
    * Transformers (Hugging Face)
    * PyTorch
    * Sentence Transformers
    * ROUGE Score
    * NLTK
    * TextStat
    * Matplotlib, Pandas
    * Ipywidgets

---

## 📊 Evaluation Metrics

To ensure a holistic comparison, the models were evaluated on the following metrics:

| Metric | Description |
| :--- | :--- |
| **ROUGE (1, 2, L)** | Measures how much of the original content is preserved in the summary. |
| **Semantic Similarity**| Checks how similar the generated summary meaning is to the original text. |
| **Readability (Flesch / Gunning Fog)** | Measures how easy it is to read and understand the summary. |
| **Processing Time** | Calculates how long each model takes to generate the summary. |

---

## 🧪 Domains Used for Testing

To make the evaluation fair and diverse, the system was tested on 10 different types of text:

1.  Biography
2.  Science & Technology
3.  Education
4.  News Articles
5.  Financial Reports
6.  Medical Texts
7.  Legal Documents
8.  Fictional Stories
9.  Product Reviews
10. Historical Texts

---

## 🚀 Getting Started

To run this project on your own, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/TextMorph.git](https://github.com/your-username/TextMorph.git)
    cd TextMorph
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    * Open the `.ipynb` file in Google Colab or Jupyter Notebook.
    * Ensure you have selected a GPU runtime for faster inference.
    * Run all the cells to launch the interactive UI.
