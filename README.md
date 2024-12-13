Overview
This repository contains the implementation of TELEClass, a hierarchical product classification system that uses LLM-enhanced core class annotation and taxonomy enrichment.
Requirements
text
torch
transformers
numpy
rank_bm25
nltk
tqdm
pandas
sentence-transformers
Installation
Clone this repository
Install dependencies:
bash
pip install -r requirements.txt
Download required NLTK data:
python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
Project Structure
model.ipynb: Main implementation notebook
AMAZON_REVIEWS.json: Input dataset (not included)
Components
Core Class Annotation (3.1)
Uses RoBERTa-large-MNLI for entailment scoring
Implements hierarchical path similarity
Limits taxonomy to 3 levels
Taxonomy Enrichment (3.2)
Uses BERT embeddings for semantic similarity
Implements BM25 scoring for term relevance
Includes category-specific term filtering
Core Class Refinement (3.3)
Uses SentenceTransformer for document embeddings
Implements confidence-based refinement
Maintains hierarchical constraints

Usage:

python
# Load data
documents = load_amazon_data("AMAZON_REVIEWS.json")

# Initialize models
core_annotator = CoreClassAnnotator()
taxonomy_enricher = TaxonomyEnricher()

# Process documents
for doc in documents:
    # Get core classes
    taxonomy = core_annotator.build_taxonomy([doc['category']])
    cores = core_annotator.get_candidates(doc['text'], taxonomy)
    
    # Enrich taxonomy
    enriched_terms = taxonomy_enricher.enrich_taxonomy(
        taxonomy=taxonomy,
        documents=[doc['text']],
        initial_cores=cores
    )
