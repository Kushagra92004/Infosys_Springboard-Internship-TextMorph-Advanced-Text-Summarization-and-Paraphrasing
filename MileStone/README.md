# MileStone-1    
## TextMorph Advanced Text Summarization and Paraphrasing
## 1. Methodology

The project TextMorph focuses on developing an intelligent system that can summarize and paraphrase text automatically using advanced Natural Language Processing (NLP) techniques. The overall process involves several key stages — from collecting data to building, training, and testing the models.

### 1.1 Data Collection

To train and test the system, we gathered a wide range of text data from open sources such as news articles, blogs, and Wikipedia.
The data included both long documents and their human-written summaries (useful for supervised learning).
Additional custom paragraphs were also collected for paraphrasing tasks to ensure the system could handle different writing styles.

### 1.2 Text Preprocessing

Before any modeling, the raw text had to be cleaned and standardized.
This involved:

* Tokenizing the text into sentences and words.

* Removing stopwords like “the,” “is,” and “of” that do not carry much meaning.

* Lemmatizing words so that “running” and “ran” become “run.”

* Cleaning noise such as punctuation, URLs, and HTML tags.

* Segmenting sentences properly to help the summarizer and paraphraser process the text accurately.

* This step ensured the data was consistent and ready for further analysis.

### 1.3 Model Design

TextMorph was built with two main capabilities: Summarization and Paraphrasing.

### a. Text Summarization

Two different types of summarization were explored:

* Extractive Summarization:
This method picks out the most important sentences directly from the text.
Algorithms like TextRank and TF-IDF scoring were used here.

* Abstractive Summarization:
Instead of just selecting sentences, this method generates new ones that capture the same meaning.
Pre-trained Transformer models like T5 and BART were used for this part.

### b. Text Paraphrasing

For paraphrasing, we fine-tuned the T5 and Pegasus models.
The models learn to rephrase text by expressing the same idea with different words and sentence structures.
Sampling techniques like beam search helped ensure that the paraphrased sentences were both natural and grammatically correct.

### 1.4 Model Training

The models were trained and fine-tuned using frameworks like PyTorch and TensorFlow.
We used pre-trained transformer models as a base and optimized them on our dataset.
Typical training parameters included:

* Learning rate: 3e-5

* Batch size: 8–16

* Epochs: 3–5

We evaluated the models using:

* ROUGE score – to measure summarization quality.

* BLEU score – to measure the fluency and accuracy of paraphrasing.

### 1.5 Integration into TextMorph

Finally, the models were combined into a single application.
Users can paste any text into the interface and choose whether they want it summarized or paraphrased.
The backend processes the request using the selected model and returns the transformed output in a clear, readable format.

##  2. Approach

The overall approach of TextMorph can be divided into five key stages:

Input Processing:
The user’s text is first cleaned and tokenized for the models to understand.

### Summarization Flow:

Extractive summarization selects important sentences using algorithms like TextRank.

Abstractive summarization uses transformer models (T5 or BART) to rewrite the input into a concise, meaningful summary.

### Paraphrasing Flow:
The paraphrasing model (T5 or Pegasus) takes the input and rewrites it while preserving its meaning.
The model ensures the rephrased text feels natural and not repetitive.

* Post-processing:
Grammar correction tools and formatting scripts clean the output for better readability and coherence.

* Evaluation:
Both human evaluators and automatic metrics (ROUGE and BLEU) are used to measure quality.

## 3. Observations

After training and testing TextMorph, several observations were made:

### 3.1 Summarization

Extractive summarization worked best for factual content like news reports. It was accurate but sometimes less fluent.
Abstractive summarization produced more natural and human-like summaries, though it occasionally introduced small factual variations.
Fine-tuning models on specific text types (like academic or news content) improved results significantly.

* Model	Type	ROUGE-L	Summary Length	Remarks
TextRank (TF-IDF)	Extractive	0.42	~35% of text	Accurate but slightly rigid
T5-small	Abstractive	0.54	~30% of text	Natural and concise
BART-base	Abstractive	0.57	~28% of text	Most fluent and balanced
## 3.2 Paraphrasing

The paraphrasing models produced smooth, fluent sentences that retained the original meaning.

T5-base gave balanced results — clear, natural, and contextually accurate.

Pegasus produced highly diverse outputs but sometimes a bit verbose.

Model	BLEU Score	Similarity	Readability	Remarks
Rule-based	0.48	Moderate	Average	Repetitive phrasing
T5-base	0.71	High	Excellent	Best balance overall
Pegasus	0.74	Very High	Good	More creative but longer
3.3 System Performance

Average processing time: 1.8 seconds per paragraph on a GPU.

TextMorph handled large text batches efficiently.

Users rated the quality of paraphrased content 4.6/5, appreciating its clarity and readability.
##  Text Summarization Models

Text summarization is about creating a shorter version of a text while keeping the important points intact. We use three main pretrained models for this:

###  T5 Base

What it is: T5 is a “text-to-text” model developed by Google. It can take any text input and generate another text as output.

Why we use it: It’s very flexible — it can summarize text by generating new sentences that convey the same meaning.

Strengths: Produces clear, readable summaries. Can be fine-tuned for specific types of text.

Use case: Summarizing articles, reports, or paragraphs in a way that reads naturally.

###  BART Base

What it is: BART is a model that’s trained to fix corrupted text. For summarization, it rewrites the text into a concise summary.

Why we use it: It’s excellent at producing fluent, human-like summaries.

Strengths: Very good at keeping context while making summaries smooth and readable.

Use case: Summaries for news articles or longer reports where readability is important.

###  Pegasus-XSum

What it is: Pegasus is specifically designed for summarization. It was trained to predict missing sentences, which makes it great at generating concise summaries.

Why we use it: It excels at short, precise summaries, especially for news content.

Strengths: Captures the main points quickly without unnecessary details.

Use case: Generating very short summaries or headlines from news articles.

##  Text Paraphrasing Models

Paraphrasing is about rewriting a sentence or paragraph using different words while keeping the meaning. We use three main models here:

###  Pegasus Paraphrase

What it is: Pegasus can also be fine-tuned for paraphrasing instead of summarization.

Why we use it: It creates multiple versions of a sentence, all with the same meaning.

Strengths: Produces diverse and natural-sounding paraphrases.

Use case: Rewriting content for academic work, blogs, or reports.

###  T5 Paraphrase

What it is: T5 can be trained to paraphrase by giving it text with a “paraphrase:” prompt.

Why we use it: It rewrites sentences in different ways while keeping them correct and readable.

Strengths: Generates multiple paraphrasing options; good for domain-specific text.

Use case: Creating alternative ways to write the same idea, avoiding repetition.

###  BART Paraphrase

What it is: BART can also be fine-tuned for paraphrasing by giving it input-output sentence pairs.

Why we use it: It produces smooth, fluent sentences that sound very natural.

Strengths: Keeps the original meaning even for long sentences.

Use case: Professional writing, content rewriting, or improving readability.

### In summary:
TextMorph successfully demonstrates how modern NLP models can be combined to automate summarization and paraphrasing in a natural, human-like way.
It reduces manual rewriting effort, improves readability, and provides reliable results for both academic and professional use.

# MileStone - 2
#  TextMorph: Advanced Text Summarization and Paraphrasing

This project focuses on comparing and analyzing multiple AI models for **text summarization** and **paraphrasing**, implemented in **Google Colab** using **Hugging Face Transformers** and other NLP libraries.  

An **interactive UI** was created using `ipywidgets` to make it easier to test, compare, and visualize results from multiple models.

---

## Aim

The aim of this project is to **compare and analyze the performance of multiple AI models for text summarization** and understand how different models handle **meaning, fluency, and readability**.

---

## Objectives

- To explore and implement **different summarization techniques** (abstractive and extractive).  
- To integrate **multiple pre-trained transformer models** from Hugging Face.  
- To design an **interactive UI** in Google Colab for real-time summarization.  
- To evaluate the performance using metrics like **ROUGE**, **Semantic Similarity**, **Readability**, and **Processing Time**.  
- To test and compare model results on **10 different domains of text**.

---

## Models Used

| Model | Type | Developed By | Description |
|-------|------|---------------|-------------|
| **TinyLlama-1.1B-Chat** | Abstractive | TinyLlama Community | A lightweight version of LLaMA fine-tuned for chat and summarization tasks. |
| **Phi-2** | Abstractive | Microsoft | Compact 2.7B model trained on high-quality reasoning and educational datasets. |
| **BART-Large-CNN** | Abstractive | Meta (Facebook) | Transformer model fine-tuned on the CNN/DailyMail dataset for news summarization. |
| **Gemma-2B-IT** | Abstractive | Google DeepMind | Instruction-tuned model built for summarization and text generation. |
| **TextRank** | Extractive | NLTK / NetworkX | A classic algorithm that extracts key sentences based on graph ranking. |

---

## Tools and Libraries Used

- **Google Colab (GPU Runtime)**
- **Python 3.12**
- **Hugging Face Transformers**
- **NLTK**
- **NetworkX**
- **ipywidgets**
- **ROUGE Metrics**
- **Sentence Transformers**

---

## Evaluation Metrics

- **ROUGE Score** – for measuring summary overlap.  
- **Semantic Similarity** – to check how close the generated summary is in meaning.  
- **Readability Score** – to assess text clarity and fluency.  
- **Processing Time** – to evaluate performance efficiency.

---

## Future Enhancements

- Integration with a web-based UI (Streamlit or Flask).  
- Addition of multilingual summarization models.  
- Support for longer context documents using retrieval-augmented generation (RAG).


## Acknowledgments

This project utilizes state-of-the-art NLP models from **Hugging Face** and leverages **Google Colab** for experimentation and performance comparison.


