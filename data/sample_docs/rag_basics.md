# Retrieval-Augmented Generation (RAG): A Quick Reference

## What RAG Is

Retrieval-Augmented Generation combines two components: a retriever that finds
relevant text from a knowledge base, and a generator (an LLM) that composes an
answer using that retrieved text as context. Instead of relying only on what
the model memorized during training, RAG grounds each answer in specific,
citable source material pulled at query time.

## Why Teams Use RAG Instead of Fine-Tuning

Fine-tuning bakes knowledge into model weights, which is expensive to update
and hard to audit. RAG keeps the knowledge base external and swappable: adding
a new document means re-embedding one file, not retraining a model. This makes
RAG the default choice when the underlying information changes frequently or
when answers need to be traceable back to a specific source for compliance or
trust reasons.

## The Core Pipeline

1. Chunking: long documents are split into smaller passages, because
   embedding models and LLM context windows both work better on focused text
   than on an entire book at once.
2. Embedding: each chunk is converted into a dense vector that captures its
   semantic meaning, so pieces of text with similar meaning end up close
   together in vector space even if they use different words.
3. Storage: those vectors are stored in a vector database (or a relational
   database with a vector extension, like Postgres with pgvector) alongside
   the original text and metadata such as the source filename.
4. Retrieval: when a user asks a question, the question itself is embedded
   with the same model, and the database returns the chunks whose vectors are
   closest to the question's vector — typically using cosine similarity.
5. Generation: the retrieved chunks are inserted into a prompt as context, and
   the LLM is instructed to answer using only that context, which reduces
   hallucination and lets the answer cite its sources.

## Common Failure Modes

Poor chunking (too large or too small) hurts retrieval precision. Using a
mismatched embedding model between ingestion and query time silently breaks
similarity search. And without an evaluation harness, it's easy to ship a RAG
system that looks fine in a demo but degrades quietly as the document
collection grows — which is why evaluation and observability are treated as
first-class parts of a production RAG system, not an afterthought.
