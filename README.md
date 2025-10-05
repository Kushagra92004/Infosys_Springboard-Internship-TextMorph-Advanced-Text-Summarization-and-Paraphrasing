# Milestone 1 – Text Summarization & Paraphrasing.

This milestone focuses on implementing and comparing pretrained language models for text summarization and paraphrasing. The goal is to evaluate model performance on sample datasets and analyze the quality of generated summaries and paraphrases through quantitative metrics and visualizations.

# Repository Structure
milestone1.ipynb  – Main Jupyter/Colab notebook (all implementation code).
art.txt, ref.txt – Sample text data files for testing summarization and paraphrasing.
/outputs/ – Folder containing generated summaries, paraphrased text, and evaluation plots after running the notebook.

# How to Run
1.Open milestone1.ipynb in Google Colab.
2.Select GPU runtime (for faster processing).
3.Run all cells sequentially from top to bottom.
4.Review the generated outputs and evaluation metrics.
5.Export or save the visualized results and generated text for analysis.

# Models Used
Summarization: t5-small,facebook/bart-large-cnn,google/pegasus-xsum
paraphrasing: attempts to use vamsi/T5_Paraphrase_Paws,ramsrigouthamg/t5_paraphraser,prithivida/parrot_paraphraser_on_T5

# Evaluation
ROUGE-1
BLEU Score

# Observation
Summarized and paraphrased text for each sample input.
Evaluation metrics such as ROUGE, BLEU, and Cosine Similarity.
Visualization plots showing performance comparisons between models.


