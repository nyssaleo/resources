# AI/ML Interview Guide: From Fundamentals to Production

## Core Concepts: What is AI/ML?

**Machine Learning** is teaching computers to learn patterns from data rather than programming explicit rules. Instead of writing "if temperature > 100°F, patient has fever," ML learns from thousands of patient records: *what patterns in vital signs, symptoms, and lab results indicate different conditions?*

**Key insight**: Traditional programming uses rules to process data → outputs. ML uses data + outputs → learns rules.

**Three Learning Types**:
- **Supervised**: Learning with labeled examples (diagnosis from labeled patient histories)
- **Unsupervised**: Finding hidden patterns without labels (discovering patient subgroups)
- **Reinforcement**: Learning through trial and reward (optimizing treatment plans)

---

## Problem Categories: The Five Core Domains

### 1. Classification
**What it predicts**: Categories or labels  
**Healthcare example**: Will this patient be readmitted within 30 days? (Yes/No)  
**How it works**: Learns decision boundaries from labeled historical data

### 2. Regression
**What it predicts**: Continuous numerical values  
**Healthcare example**: What will this patient's blood pressure be in 6 months? (120 mmHg)  
**How it works**: Learns mathematical relationships between input features and numeric outcomes

### 3. Clustering
**What it discovers**: Natural groupings in data  
**Healthcare example**: Segment patients into risk groups without predefined labels  
**How it works**: Measures similarity and groups similar data points together

### 4. Natural Language Processing (NLP)
**What it processes**: Human language  
**Healthcare example**: Extract medication names and dosages from doctor's notes  
**How it works**: Converts text to numerical representations, then applies ML

### 5. Computer Vision
**What it analyzes**: Images and video  
**Healthcare example**: Detect tumors in chest X-rays  
**How it works**: Learns hierarchical visual patterns from pixels to concepts

---

## Problem-Solution Mapping: The Decision Framework

**Three Questions to Choose Your Approach**:

1. **What data do I have?**
   - Labeled? → Supervised learning (Classification/Regression)
   - Unlabeled? → Unsupervised learning (Clustering)
   - Sequential decisions? → Reinforcement learning

2. **What am I predicting?**
   - Category/Label? → Classification (Random Forest, Neural Networks)
   - Number/Value? → Regression (Linear Regression, XGBoost)
   - Unknown structure? → Clustering (K-Means, DBSCAN)

3. **How complex are the patterns?**
   - Linear relationships? → Linear models (fast, interpretable)
   - Complex non-linear? → Tree-based or Neural Networks
   - Images/Text/Speech? → Deep Learning (CNNs, Transformers)

**Key Trade-off**: Interpretability vs. Accuracy. Linear regression explains *why*, deep neural networks predict *what* more accurately.

---

## Methodologies: How Core Algorithms Work

### Linear Regression
**Concept**: Draw the best-fit line through data points  
**Example**: Predict systolic blood pressure from age: `BP = 110 + (0.5 × age)`  
**When to use**: Simple relationships, need to explain feature importance

### Decision Trees / Random Forest
**Concept**: Series of yes/no questions leading to predictions  
**Example**: "Age > 65? → Yes → Smoker? → Yes → High risk"  
**When to use**: Non-linear patterns, need interpretability, mixed data types

### Neural Networks / Deep Learning
**Concept**: Layers of mathematical transformations that learn hierarchical patterns  
**Example**: Layer 1 detects edges in X-ray → Layer 2 detects shapes → Layer 3 detects tumors  
**When to use**: Complex patterns, large datasets, images/text/audio

### K-Means Clustering
**Concept**: Group data into K clusters by minimizing distance to cluster centers  
**Example**: Segment patients into 4 risk tiers based on 20 health metrics  
**When to use**: Customer segmentation, anomaly detection, data exploration

### Support Vector Machines (SVM)
**Concept**: Find the optimal boundary that maximally separates different classes  
**Example**: Separate healthy vs. diseased patients in high-dimensional lab test space  
**When to use**: Small to medium datasets, clear separation needed, binary classification

---

## Real-World Applications

**Healthcare**: Predict sepsis 6 hours before onset (Johns Hopkins) | Detect diabetic retinopathy from eye scans (Google Health)

**Finance**: Credit scoring (XGBoost on transaction history) | Fraud detection (anomaly detection on spending patterns)

**Retail**: Demand forecasting (time-series regression) | Product recommendations (collaborative filtering)

**Manufacturing**: Predictive maintenance (classification: will machine fail?) | Quality control (computer vision defect detection)

---

## Advanced Techniques: Modern AI Stack

### Embeddings
**Concept**: Convert text/images into numerical vectors that capture semantic meaning  
**Example**: "chest pain" and "angina" get similar vector representations despite different words  
**How it works**: Neural networks trained on massive datasets learn to encode meaning into numbers (typically 768-1536 dimensions)  
**Why it matters**: Enables similarity search, clustering, and feeds into other ML models

### Semantic Search
**Concept**: Search by meaning, not keywords  
**Example**: Query "heart attack symptoms" returns documents about "myocardial infarction" and "cardiac arrest"  
**How it works**: Convert query and documents to embeddings → measure cosine similarity → return closest matches  
**Traditional vs. Semantic**: BM25 matches words; embeddings match concepts

### RAG (Retrieval-Augmented Generation)
**Concept**: Give LLMs access to external knowledge before generating responses  
**Example**: Medical chatbot retrieves latest treatment guidelines from database before answering "What's the protocol for severe COVID?"  
**How it works**: Question → Semantic search retrieves relevant documents → LLM generates answer using retrieved context  
**Why it's powerful**: Keeps LLMs current without retraining; reduces hallucinations; adds citations

**RAG Pipeline**: 
1. Chunk documents into passages
2. Generate embeddings for each chunk
3. Store in vector database (Pinecone, Weaviate, ChromaDB)
4. User query → retrieve top-K similar chunks
5. LLM generates answer with retrieved context

### Vector Databases
**Purpose**: Store and rapidly search millions of embeddings  
**Key algorithms**: HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index)  
**Use case**: Power semantic search, RAG, recommendation systems

---

## Industry Applications: LLMs in Practice

**Customer Support**: Chatbots answer FAQs (RAG), classify tickets (fine-tuned BERT), route to specialists (embeddings + classification)

**Legal**: Contract analysis (NLP entity extraction), precedent search (semantic search across case law), clause recommendation (RAG)

**Software Engineering**: Code completion (GitHub Copilot uses Codex), bug detection (pattern recognition), documentation generation (GPT-4)

**Content Creation**: Article generation (GPT-4), image generation (DALL-E, Stable Diffusion), video synthesis (combining vision + language models)

**Enterprise Search**: "Find all projects related to sustainability in Q3" → semantic search across documents, emails, databases

---

## Interview Power Phrases

✓ "I'd use classification for this because we're predicting discrete categories, not continuous values"  
✓ "RAG is ideal here because we need current information without retraining the entire model"  
✓ "Embeddings let us capture semantic similarity—similar meanings cluster together in vector space"  
✓ "The trade-off is interpretability versus accuracy: linear models explain decisions, neural networks optimize performance"  
✓ "We'd validate this with train-test split and cross-validation to ensure it generalizes to unseen data"

---

**Core Principle**: Start simple (linear regression, basic classification), add complexity only when simple models fail. The best model is the simplest one that meets your accuracy requirements.

