# Milestone 1 – TextMorph Advanced Text Summarization and Paraphrasing
* This milestone focuses on implementing and comparing pretrained language models for text summarization and paraphrasing. The goal is to evaluate model performance on sample datasets and analyze the quality of generated summaries and paraphrases through quantitative metrics and visualizations.

## Structure
* milestone1.ipynb  – Main Jupyter/Colab notebook (all implementation code).
* art.txt, ref.txt – Sample text data files for testing summarization and paraphrasing.
* /outputs/ – Folder containing generated summaries, paraphrased text, and evaluation plots after running the notebook.

## How to Run
1.Open milestone1.ipynb in Google Colab.

2.Select GPU runtime (for faster processing).

3.Run all cells sequentially from top to bottom.

4.Review the generated outputs and evaluation metrics.

5.Export or save the visualized results and generated text for analysis.

## Models Used
* Summarization: t5-small,facebook/bart-large-cnn,google/pegasus-xsum
* paraphrasing: attempts to use vamsi/T5_Paraphrase_Paws,ramsrigouthamg/t5_paraphraser,prithivida/parrot_paraphraser_on_T5

## Evaluation
* ROUGE-1

* BLEU Score

## Observation
* Summarized and paraphrased text for each sample input.
* Evaluation metrics such as ROUGE, BLEU, and Cosine Similarity.
* Visualization plots showing performance comparisons between models.

## Text Summarization:
Text summarization means creating a shorter version of a text that still keeps the main ideas and important information from the original.

It’s like reading a long article or story and then writing a brief summary that tells what it’s mainly about.

There are two main types of text summarization:

Extractive summarization:

Picks out the most important sentences or phrases directly from the text.

Example: Highlighting key sentences from a news article.

Abstractive summarization:

Rewrites the text in new words, like how a human would summarize.

It uses understanding of meaning, not just sentence picking.



