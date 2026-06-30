# Building a RAG Chatbot Over Confluence Using AWS S3 Vectors and BGE Embeddings

*How we built an internal engineering knowledge assistant that lets our team query architecture decisions, pipeline designs, and system documentation in natural language — instead of digging through hundreds of Confluence pages.*

---

## The Problem

Our engineering team maintains hundreds of Confluence pages covering product architectures, data pipeline designs, infrastructure decisions, and operational runbooks. Finding the right information meant:

- Manually searching through nested Confluence spaces
- Asking the one person who "knows where that doc is"
- Re-discovering decisions that were already made months ago

The institutional knowledge was there — it was just trapped in a format that didn't scale.

## The Solution: A RAG-Powered Engineering Chatbot

We built **Engineering Compass** — a Retrieval-Augmented Generation (RAG) chatbot that ingests our entire Confluence workspace and lets engineers ask questions in natural language.

Ask: *"What database does the backend service use and why?"*
Get: A grounded answer with citations to the exact Confluence page and section.

---

## Architecture Overview

The system is split into two decoupled pipelines — **ingestion** (batch, offline) and **retrieval + generation** (real-time, per-query). This separation lets us re-index Confluence content independently of serving, and scale each pipeline on its own terms.

### Ingestion Pipeline

Confluence pages flow through four stages before becoming searchable vectors:

```
Confluence API ──→ HTML Parser ──→ Smart Chunker ──→ Embedder ──→ S3 Vectors
                   (BeautifulSoup)   (text/tables/    (BGE-large,   (cosine index,
                                      diagrams)        1024-dim)     float32)
```

1. **Fetch** — Pull raw HTML (`body.storage`) from Confluence Cloud via REST API, including page metadata and image attachments.
2. **Parse** — Strip Confluence macros and extract clean content using BeautifulSoup, preserving heading hierarchy.
3. **Chunk** — Split content by type: text sections respect heading boundaries (800 chars, 100 overlap), tables convert to structured key-value format, and diagrams pass through LLaVA (13B, local via Ollama) to produce text descriptions.
4. **Embed & Store** — Each chunk gets encoded with BGE-large-en-v1.5 using the document prefix, normalized, and written to an S3 Vectors index with metadata (page title, URL, heading, content type).

### Retrieval + Generation Pipeline

At query time, the path from question to grounded answer takes ~1 second:

```
User Query ──→ Query Embedding ──→ S3 Vectors Search ──→ Top-K Chunks ──→ LLM ──→ Answer
               (BGE-large,          (cosine similarity)   (k=5)           (Bedrock
                query prefix)                                              Claude 3 Haiku)
```

1. **Embed query** — Encode the user's question with the `"Represent this question: "` prefix (asymmetric encoding improves recall over symmetric approaches).
2. **Vector search** — Query S3 Vectors for the top-5 most similar chunks using cosine distance. Metadata filters can narrow results by content type or Confluence space.
3. **Generate** — Pass retrieved chunks as context to Claude 3 Haiku via Amazon Bedrock. The prompt instructs the model to answer only from the provided context and cite sources.
4. **Return** — The response includes the generated answer plus source citations (page title, URL, section heading) so engineers can verify and dig deeper.

### Why This Separation Matters

- **Ingestion is batch and fault-tolerant** — a failed page doesn't block the rest; re-runs are idempotent (keyed by page ID + chunk index).
- **Retrieval is stateless** — the FastAPI serving layer holds no state beyond the embedding model in memory. Horizontal scaling is trivial.
- **Each component is independently replaceable** — swap the embedding model, change the vector store, or switch the LLM without touching the other pieces.

---

## Step 1: Fetching Pages from Confluence

We use the `atlassian-python-api` library to connect to Confluence Cloud and pull pages with their full HTML body:

```python
from atlassian import Confluence

def get_confluence_client() -> Confluence:
    return Confluence(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_EMAIL"),
        password=os.getenv("CONFLUENCE_API_TOKEN"),
        cloud=True,
    )

def fetch_page(confluence: Confluence, page_id: str) -> dict:
    return confluence.get_page_by_id(
        page_id=page_id,
        expand="body.storage,version",
    )
```

We fetch pages by space, pulling the raw HTML (`body.storage`) which contains rich content — text, tables, embedded images, and macros.

---

## Step 2: Smart Chunking (Text + Tables + Diagrams)

This is where most RAG pipelines fail. Naive chunking (split every 500 chars) destroys context. We built a **multi-modal chunker** that handles three content types differently:

### Text Sections
Split by heading structure, respecting section boundaries. Each chunk keeps its heading for context:

```python
def chunk_text_sections(text: str, max_chunk_size: int = 800, overlap: int = 100):
    sections = split_into_sections(text)
    chunks = []
    
    for section in sections:
        content = section["content"]
        if len(content) <= max_chunk_size:
            chunks.append({
                "heading": section["heading"],
                "content": content,
                "content_type": "text",
            })
        else:
            # Split long sections by paragraphs with overlap
            # ... (paragraph-level splitting with 100-char overlap)
    return chunks
```

### Tables
Confluence is full of comparison tables and config matrices. We extract them as structured text:

```python
def extract_tables_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    
    for table in tables:
        # Find preceding heading for context
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        
        # Convert rows to "Header: Value | Header: Value" format
        # Large tables get summarized; full content kept in metadata
```

### Architecture Diagrams — The Hidden Knowledge Problem

Here's something most RAG tutorials skip: engineering documentation is full of **architecture diagrams**. In our Confluence workspace, critical information — service relationships, data flows, protocol choices, infrastructure topology — lives *only* in diagrams, not in the surrounding text.

A standard text-only RAG pipeline would completely miss this knowledge. If someone asks "how does data flow from ingestion to the analytics layer?", the answer might exist only in a diagram that was never converted to text.

**Our solution: LLaVA (Large Language and Vision Assistant) via Ollama.**

We run [LLaVA 13B](https://ollama.com/library/llava) locally through Ollama. For every image attachment on a Confluence page, we:

1. Download the image from Confluence's attachment API
2. Feed it to LLaVA with a detailed prompt
3. Get back a rich text description of the diagram
4. Chunk and embed that description alongside the regular text

```python
import ollama

def describe_image_with_llava(image_path: str) -> str:
    response = ollama.chat(
        model="llava:13b",
        messages=[{
            "role": "user",
            "content": (
                "Describe this architecture diagram in detail. "
                "Include all components, connections, data flows, protocols, "
                "and any labels or annotations visible. "
                "Be specific about the direction of arrows and relationships."
            ),
            "images": [image_path],
        }],
    )
    return response["message"]["content"]
```

**Why Ollama + LLaVA over cloud vision APIs (GPT-4V, Claude Vision)?**

| Consideration | Ollama + LLaVA | Cloud Vision API |
|---|---|---|
| **Data privacy** | Runs 100% local — diagrams never leave the machine | Diagrams sent to external servers |
| **Cost** | Free (one-time model download, ~8GB) | $0.01–0.03 per image |
| **Latency** | ~10-15s per image (acceptable for batch ingestion) | ~3-5s per image |
| **Quality** | Good enough for structured diagrams with labels | Slightly better for complex visuals |
| **Offline** | Works without internet | Requires API connectivity |

For our use case — batch-processing internal architecture diagrams during ingestion (not real-time) — the privacy and cost tradeoffs made LLaVA the clear winner. Our diagrams contain internal service names, IP ranges, and infrastructure details that we don't want leaving our network.

The prompt engineering matters here. A generic "describe this image" gives weak results. Being specific about what to extract (components, connections, data flows, direction of arrows) gives descriptions that are actually useful for retrieval.

**Example output** for a typical microservices architecture diagram:

> "The diagram shows a data pipeline with three main layers. The ingestion layer on the left contains Kinesis Data Streams receiving events from multiple producers. Arrows flow right into a processing layer with AWS ECS containers running Spark jobs. The processed data flows into a storage layer with S3 (raw), Redshift (analytics), and DynamoDB (serving). A monitoring component at the top connects to all layers via CloudWatch metrics..."

This text gets chunked and embedded just like any other content — meaning vector search can now surface answers that were previously locked inside images.

### Architecture Diagrams
Engineering docs are full of architecture diagrams. We download image attachments from Confluence and use **LLaVA (13B)** locally via Ollama to generate text descriptions:

```python
def describe_image_with_llava(image_path: str) -> str:
    response = ollama.chat(
        model="llava:13b",
        messages=[{
            "role": "user",
            "content": (
                "Describe this architecture diagram in detail. "
                "Include all components, connections, data flows, protocols, "
                "and any labels or annotations visible."
            ),
            "images": [image_path],
        }],
    )
    return response["message"]["content"]
```

This means when someone asks "what's the architecture of system X?", the chatbot can reference diagram descriptions alongside text — even if the answer was only in an image.

---

## Step 3: Embedding with BGE-large-en-v1.5

We chose [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for embeddings. Why:

- **Top-5 on MTEB** (Massive Text Embedding Benchmark) for retrieval tasks
- **1024 dimensions** — good balance of quality vs. storage cost
- **Instruction-based** — separate prefixes for documents vs. queries improves matching
- **Self-hosted** — no data leaves our infrastructure (critical for internal docs)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Documents get this prefix (tells the model this is a document to be retrieved)
texts = ["Represent this document: " + chunk["content"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,   # Required for cosine similarity
    show_progress_bar=True,
    batch_size=8,
)
# Output shape: (num_chunks, 1024)
```

At query time, we prefix with `"Represent this question: "` — this asymmetric encoding significantly improves retrieval quality for question-answering use cases.

---

## Step 4: Storing in AWS S3 Vectors

We use **AWS S3 Vectors** — a serverless vector store built into S3. No infrastructure to manage, no separate vector DB to operate.

```python
import boto3

client = boto3.Session(
    region_name=os.getenv("AWS_REGION")
).client("s3vectors")

# Create the index (one-time)
client.create_index(
    vectorBucketName=os.getenv("VECTOR_BUCKET_NAME"),
    indexName="confluence-docs",
    dataType="float32",
    dimension=1024,
    distanceMetric="cosine",
)

# Upload vectors with metadata
vectors = [{
    "key": f"page-{page_id}-chunk-{chunk['chunk_index']}",
    "data": {"float32": chunk["embedding"]},
    "metadata": {
        "page_title": metadata["page_title"],
        "page_url": metadata["page_url"],
        "heading": chunk["heading"],
        "content_type": chunk["content_type"],
        "text": chunk["content"][:1000],  # Store text for retrieval display
    },
} for chunk in embedded_chunks]

# Batch upload
client.put_vectors(
    vectorBucketName=os.getenv("VECTOR_BUCKET_NAME"),
    indexName="confluence-docs",
    vectors=vectors,
)
```

**Why S3 Vectors over Pinecone/Weaviate/pgvector?**
- Zero infra management — no clusters, no scaling config
- Native AWS IAM integration — same auth as everything else
- Pay per storage + query — no idle costs
- Metadata filtering built-in — filter by page, content type, etc.

---

## Step 5: Query Pipeline (FastAPI)

The serving layer is a lightweight FastAPI app:

```python
@app.post("/ask")
async def ask_question(query: str):
    # 1. Embed the query (with question prefix)
    query_embedding = model.encode(
        "Represent this question: " + query,
        normalize_embeddings=True,
    )
    
    # 2. Search S3 Vectors for top-K similar chunks
    results = s3vectors_client.query_vectors(
        vectorBucketName=os.getenv("VECTOR_BUCKET_NAME"),
        indexName=os.getenv("VECTOR_INDEX_NAME"),
        queryVector={"float32": query_embedding.tolist()},
        topK=5,
    )
    
    # 3. Build context from retrieved chunks
    context = "\n\n".join([r["metadata"]["text"] for r in results["vectors"]])
    
    # 4. Generate answer with Claude (via Bedrock)
    answer = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku",
        body={
            "prompt": f"Based on this context:\n{context}\n\nAnswer: {query}",
            "max_tokens": 500,
        }
    )
    
    return {"answer": answer, "sources": results["vectors"]}
```

---

## Results

- **Coverage:** All engineering documentation searchable through a single interface
- **Content types:** Text, tables, and architecture diagrams — all queryable
- **Latency:** Sub-second responses (embedding + vector search + LLM generation)
- **Cost:** Minimal — S3 Vectors is serverless, BGE runs locally, only LLM calls cost money
- **Maintenance:** Confluence pages get re-indexed on update (webhook trigger)

---

## Key Decisions & Tradeoffs

| Decision | Why |
|----------|-----|
| BGE-large over OpenAI embeddings | Self-hosted, no data leakage, top MTEB performance |
| S3 Vectors over Pinecone | Zero ops, native AWS, no vendor lock-in |
| LLaVA for diagrams | Keeps image understanding local, no API costs |
| 800-char chunks with 100-char overlap | Preserves section context without losing precision |
| Cosine distance metric | Standard for normalized text embeddings |
| Metadata in vectors | Enables filtered search (by page, content type) |

---

## What's Next

- **Incremental sync** — webhook-triggered re-indexing when pages are updated
- **Multi-space support** — expanding beyond engineering docs to product and ops
- **Slack integration** — ask questions directly in Slack channels
- **Feedback loop** — track which answers get thumbs-up to improve retrieval

---

## Tech Stack

- **Ingestion:** Python, atlassian-python-api, BeautifulSoup
- **Chunking:** Custom multi-modal (text + tables + images via LLaVA/Ollama)
- **Embeddings:** BAAI/bge-large-en-v1.5 (sentence-transformers)
- **Vector Store:** AWS S3 Vectors (cosine, 1024-dim, float32)
- **Serving:** FastAPI + Uvicorn
- **LLM:** Amazon Bedrock (Claude 3 Haiku)
- **Infra:** AWS CDK, S3, IAM

---

*If you're building something similar, the key insight is: don't treat all content the same. Tables, diagrams, and text each need different chunking strategies. A naive `text.split()` approach will give you garbage retrieval on real-world enterprise docs.*

---

**Prashant Kumar Singh** — Lead Data Engineer
[LinkedIn](https://linkedin.com/in/pr-ashant-singh) | [GitHub](https://github.com/pr-ashant-singh)
