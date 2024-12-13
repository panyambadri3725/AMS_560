# TELEClass: Taxonomy-Enhanced Large-scale E-commerce Product Classification

## Overview
TELEClass is a hierarchical product classification system that combines LLM-enhanced core class annotation with taxonomy enrichment to improve e-commerce product categorization.

## Requirements
```bash
torch
transformers
numpy
rank_bm25
nltk
tqdm
pandas
sentence-transformers
scikit-learn
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/TELEClass.git
cd TELEClass
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
Run the following in a Python shell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Components

### 1. Core Class Annotation (3.1)
- Uses **RoBERTa-large-MNLI** for entailment scoring.
- Implements hierarchical path similarity with temperature scaling.
- Limits taxonomy to **3 levels** for efficiency.
- Features top-down candidate selection.

### 2. Taxonomy Enrichment (3.2)
- Uses **BERT embeddings** with GPU acceleration.
- Implements batch processing for efficiency.
- Features embedding caching mechanism.
- Uses **BM25 scoring** for term relevance.
- Combines popularity, distinctiveness, and semantic similarity scores.

### 3. Core Class Refinement (3.3)
- Uses **SentenceTransformer** for document embeddings.
- Implements confidence-based refinement.
- Maintains hierarchical constraints.
- Features batch processing for GPU utilization.

## Usage

### Example Code
```python
# Load and preprocess data
documents = load_amazon_data("AMAZON_REVIEWS.json")

# Initialize models
core_annotator = CoreClassAnnotator()
taxonomy_enricher = TaxonomyEnricher()

# Process documents
for doc in documents:
    # Build taxonomy and get core classes
    taxonomy = core_annotator.build_taxonomy([doc['category']])
    cores = core_annotator.get_candidates(doc['text'], taxonomy)
    
    # Enrich taxonomy
    enriched_terms = taxonomy_enricher.enrich_taxonomy(
        taxonomy=taxonomy,
        documents=[doc['text']],
        initial_cores=cores
    )
```

## Data Format

Input JSON should have the following structure:

```json
{
    "title": "Product title",
    "description": "Product description",
    "category": ["Level1", "Level2", "Level3"],
    "reviews": [{"text": "Review text"}]
}
```

## GPU Optimization

The implementation includes several optimizations for GPU usage:
- **Batch processing** for embeddings.
- **Embedding caching** to avoid redundant calculations.
- Efficient similarity calculations.
- Minimized CPU-GPU transfers.
- **Vectorized operations** where possible.

