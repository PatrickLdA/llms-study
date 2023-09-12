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

[Notebook 03_vectorstores_and_embeddings](notebooks/03_vectorstores_and_embeddings.ipynb)

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

[Notebook 04_retrieval](notebooks/04_retrieval.ipynb)


# Question Answering

![Alt text](images/qa_chain.png)

To deal with the problem of short context window, there are 3 methods:

- Map_reduce: each chunk is sent to the model and the answers are sent together to a final response. Slower
- Refine
- Map_rerank

![Alt text](images/methods_context_window.png)

[Notebook 05_question_answering.ipynb](notebooks/05_question_answering.ipynb)


# Chat

Similar as previous content, but with a `Chat history`

![Alt text](images/chat_llm.png)

[Notebook 06_chat.ipynb](notebooks/06_chat.ipynb)