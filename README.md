# Infosys_Springboard-Internship-TextMorph-Advanced-Text-Summarization-and-Paraphrasing
TextMorph is an advanced natural language processing (NLP) system designed to automatically summarize and paraphrase large volumes of text efficiently and intelligently. It combines modern AI language models with linguistic analysis to generate concise, coherent, and contextually accurate representations of source content.
Milestone 1 – Text Summarization & Paraphrasing
Overview

This milestone focuses on implementing and comparing pretrained language models for text summarization and paraphrasing. The goal is to evaluate model performance on sample datasets and analyze the quality of generated summaries and paraphrases through quantitative metrics and visualizations.

Repository Structure

milestone1.ipynb – Main Jupyter/Colab notebook (all implementation code).

art.txt, ref.txt – Sample text data files for testing summarization and paraphrasing.

/outputs/ – Folder containing generated summaries, paraphrased text, and evaluation plots after running the notebook.

How to Run

Open milestone1.ipynb in Google Colab.

Select GPU runtime (for faster processing).

Run all cells sequentially from top to bottom.

Review the generated outputs and evaluation metrics.

Export or save the visualized results and generated text for analysis.

Models Used

Summarization Models:

t5-small

facebook/bart-large-cnn

google/pegasus-xsum

(Other tested models can be added here)

Paraphrasing Models:

vamsi/T5_Paraphrase_Paws

ramsrigouthamg/t5_paraphraser

prithivida/parrot_paraphraser_on_T5

Outputs

Summarized and paraphrased text for each sample input.

Evaluation metrics such as ROUGE, BLEU, and Cosine Similarity.

Visualization plots showing performance comparisons between models.

Objective

To identify the most efficient and semantically accurate pretrained models for summarization and paraphrasing tasks, forming the foundation for further improvement in later project milestones.
