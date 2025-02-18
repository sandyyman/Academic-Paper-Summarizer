# Academic Paper Summarizer

An AI-powered academic paper analysis tool that helps researchers quickly understand and analyze research papers using Groq LLM and vector similarity search.

## Features

- **Paper Search**: Search for academic papers on ArXiv by topic
- **Smart Summaries**: Generate concise summaries of research papers
- **Paper Analysis**: Extract key concepts and technical details
- **Similar Papers**: Find related research using vector similarity

## Quick Start

### Prerequisites
bash
Python 3.8+
pip (Python package manager)
*env*

GROQ_API_KEY=your_groq_api_key_here
ARXIV_EMAIL=your_email@example.com



## Common Issues & Fixes

### Embedding Dimension Mismatch
If you see errors like: Error adding paper: Embedding dimension 76 does not match collection dimensionality 89


Fix by clearing the vector store:
1. Stop the application
2. Delete the `data` directory:
   bash
   rm -rf data
3. Restart the application

### Installation Issues
If you encounter dependency errors:
bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Technologies Used

## Core Technologies

### 1. Frontend & Interface
- **Streamlit** (v1.24.0+)
### 2. Language Models & AI
- **Groq LLM**
### 3. Vector Storage & Similarity Search
- **ChromaDB** (v0.4.22+)
### 4. Paper Retrieval & Analysis
- **ArXiv API**
### 5. Text Processing
- **scikit-learn**

## Technical Details

### Vector Store Configuration
The application uses ChromaDB for vector storage with:
- Fixed embedding dimension: 512
- Cosine similarity metric
- TF-IDF vectorization

### LLM Integration
- Uses Groq's LLM via LangChain
- Model: mixtral-8x7b-32768
- Temperature: 0.7
- Max tokens: 4096

## Snapshot

![Screenshot (338)](https://github.com/user-attachments/assets/f86b7267-5dd5-4601-bf8c-0d42bddc5f28)

![Screenshot (339)](https://github.com/user-attachments/assets/788e9d80-9ac7-4efb-b521-0d2f0982f519)

![Screenshot (340)](https://github.com/user-attachments/assets/c06d7c53-0af6-4142-a4a9-65401ea0c96d)

## 🔍 Troubleshooting Guide

### Vector Store Issues
1. **Dimension Mismatch**
   - Clear the data directory
   - Ensure consistent vectorizer configuration
   - Restart application

2. **ChromaDB Errors**
   - Check data directory permissions
   - Verify ChromaDB installation
   - Update ChromaDB: `pip install --upgrade chromadb`

3. **Memory Issues**
   - Reduce max_results parameter
   - Clear browser cache
   - Restart application

### API Issues
1. **Groq API**
   - Verify API key in .env
   - Check API rate limits
   - Monitor API response errors

2. **ArXiv API**
   - Check internet connection
   - Verify email in .env
   - Handle request timeouts
