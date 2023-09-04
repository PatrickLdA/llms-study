https://learn.deeplearning.ai/langchain-chat-with-your-data

> LangChain and LLMs are a way to chat with your data

# Document loading

80 different types of document loaders to access

- Websites
- DBs
- Youtube
- arXiv...
- PDF
- HTML
- Json
- Word, PPT...

[Notebook 1](notebooks/01_document_loading.ipynb)

# Document splitting

Happens after loading the data

![Alt text](images/splitting.png)

To avoid the loss of information from one chunk to the other, a **chunk overlap** can be used

![Alt text](images/chunk_overlap.png)

Chunking can happen in different ways

![Alt text](images/splitting_methods.png)

[Notebook 2](notebooks/02_document_splitting.ipynb)

# Vectorstores and embeddings

![Alt text](images/vectorstore.png)

# Retrieval

## Maximum Marginal Relevance (MMR)

You may not always want to choose the most similar responses

- Diversity of information

How

- Query the vector store
- `fetch_k` most similar responses
- Within those responses choose the `k` most diverse

## LLM Aided retrieval

![Alt text](images/llm_aided_retrieval.png)

## Compression

![Alt text](images/compression.png)

[Notebook](https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/5/retrieval)