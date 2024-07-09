# RAG Application

This project demonstrates an end-to-end implementation of a Retrieval-Augmented Generation (RAG) system using LangChain. The system utilizes document embeddings, a vector store, and a language model to answer queries based on provided documents.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Retrieval-Augmented Generation (RAG) system is designed to answer questions by retrieving relevant context from a document database and generating responses using a language model. This project leverages LangChain's capabilities to build, query, and test the RAG system.

## Features

- Load and process PDF documents
- Split documents into manageable chunks
- Store document chunks in a Chroma vector store
- Query the vector store to retrieve relevant context
- Use a language model to generate answers based on the retrieved context
- Evaluate the system's responses against expected answers

## Setup

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aerospacerr/langchain-rag-llama3.git
    cd langchain-rag-llama3
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
