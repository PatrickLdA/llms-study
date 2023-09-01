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

