## Chatbot based on RAG built on Gradio

[Deployed On HF Space] (https://huggingface.co/spaces/gauravkumarML/ragbot-gradio)

chatbot-rag is a Retrieval-Augmented Generation (RAG) chatbot implementation intended to combine vector-based retrieval (documents / knowledge base) with a generative LLM to answer queries, perform tasks, or compete in RAG-style challenge tasks. The core script appears to be competition_rag_app.py.

This repo is a compact starting point to:

ingest documents and create embeddings,

store/retrieve using a vector store,

assemble retrieved context + prompt for the LLM,

generate final answers.

[Demo Screengrab] (assets/demo.png)
