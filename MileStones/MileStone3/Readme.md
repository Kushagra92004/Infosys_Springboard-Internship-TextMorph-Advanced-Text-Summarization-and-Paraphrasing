# TextMorph: Text Summarization and Paraphrasing Platform

## Project Overview
TextMorph is an NLP-powered web application designed for text summarization, paraphrasing, and readability analysis.  
Developed using Streamlit, it enables users to transform and understand text through intelligent summarization and complexity-based paraphrasing — all within a secure and intuitive interface.

---

## Key Features

### User Authentication & Security
- Secure JWT-based authentication system  
- Role-based access with user profiles  
- Reading preferences and content type management  
- Encrypted login and registration interface  

### Readability Analysis Dashboard
- Real-time readability scoring using multiple indices:
  - Flesch-Kincaid Readability Score  
  - Gunning Fog Index  
  - SMOG Index  
- Color-coded complexity visualization  
- Drag-and-drop file upload  
- Comprehensive text metrics display  

### Multi-Level Summarization
- Integrated models: PEGASUS, BART, and FLAN-T5  
- Adjustable summary lengths: Short, Medium, Long  
- Side-by-side comparison of original and summarized text  
- Summary quality evaluation using ROUGE metrics  
- Word count tracking and compression ratio calculation  

### Complexity-Based Paraphrasing
- Multiple complexity levels: Beginner, Intermediate, Advanced  
- Readability grade-level control  
- Style-based paraphrasing options  
- Original vs paraphrased text comparison  

---

## Technical Implementation

### Models and Tools
- PEGASUS – Abstractive summarization  
- BART – Sequence-to-sequence text generation  
- FLAN-T5 – Versatile text-to-text transformer  
- NLTK – Readability analysis and text preprocessing  

### Security and Deployment
- JWT for secure session management  
- Ngrok for tunneling and public access  
- Streamlit-based responsive web interface  
- Secure file upload and session handling  

---

## Technical Challenges and Solutions

| Area | Challenge | Solution |
|------|------------|-----------|
| Authentication | Token expiration and session management | Implemented JWT refresh and secure session state in Streamlit |
| Model Integration | High memory usage from PEGASUS and BART | Added model caching and selective loading mechanisms |
| Performance | Slow inference on CPU | Optimized batch processing and added progress indicators |
| Ngrok Stability | Tunnel drops and port conflicts | Implemented auto-reconnect and dynamic port allocation |
| Readability Accuracy | Noise from special characters | Added robust text cleaning and normalization |
| UI/UX Layout | Streamlit’s limited layout control | Utilized columns, expanders, and custom CSS for structure |

---

## Contributors

| Name | Contribution | GitHub |
|------|---------------|--------|
| Lithicka | Implemented readability analysis using NLTK indices, developed visual complexity indicators, and created real-time scoring and metrics display | [GitHub](https://github.com/Lithicka-G) |
| Aakanksha Jadhav | Implemented JWT-based authentication, user profiles, and session management | [GitHub](https://github.com/aakankshajadav) |
| Kushgra Hada | Designed and developed the UI/UX, dashboard layout, and navigation | [GitHub](https://github.com/kushgrahada) |
| Sherwin Jonathan | Integrated summarization and paraphrasing models, optimized performance | [GitHub](https://github.com/sherwinjonathan) |

---

## Installation and Setup

1. **Clone the Repository**  
   Clone the TextMorph repository from GitHub to your local machine.

2. **Install Dependencies**  
   Install all required Python packages using pip. Ensure your system has **Python 3.8 or higher** before installing.

3. **Configure Environment Variables**  
   Set up the following environment variables:  
   - `JWT_SECRET_KEY` – for authentication token generation  
   - `NGROK_AUTH_TOKEN` – for secure tunneling and public access  
   These are essential for the application’s security and proper functionality.

4. **Launch the Application**  
   Start the Streamlit application to run the local development server. The app will open in your web browser, allowing you to access TextMorph.

5. **Register or Log In**  
   Create a new account or log in to start using all the features of TextMorph.

---

## Usage
1. Register or log in to access all features.  
2. Upload or paste text directly into the interface.  
3. View real-time readability scores and metrics.  
4. Generate concise summaries using the integrated models.  
5. Paraphrase text to adjust style and complexity levels.  

---

## Technology Stack
- **Frontend:** Streamlit  
- **Authentication:** JWT Tokens  
- **NLP Models:** PEGASUS, BART, FLAN-T5  
- **Text Analysis:** NLTK  
- **Security and Access:** Ngrok  
- **Deployment:** Streamlit Cloud / Local  

---
## Team 1
TextMorph was developed by a collaborative team of AI and software enthusiasts committed to building an intuitive and secure NLP platform.  
The team combined expertise in **NLP modeling, web development, UI/UX design, and security** to create a seamless experience for text summarization and paraphrasing.
