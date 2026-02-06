"""
Product Review RAG System
Combines embedding models with chunking strategies for product review Q&A
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from rapidfuzz import fuzz, process, utils


@dataclass
class ReviewChunk:
    """Represents a single review chunk with metadata"""
    product_id: str
    product_name: str
    category: str
    rating: float
    review_title: str
    review_content: str
    user_name: str
    chunk_text: str
    chunk_type: str  # 'product_summary' or 'individual_review'


class ProductReviewRAG:
    """RAG system for product review analysis"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the RAG system
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[ReviewChunk] = []
        self.embeddings: np.ndarray = None
        
    def create_chunks(self, df: pd.DataFrame) -> List[ReviewChunk]:
        """
        Create chunks from product review data using hybrid strategy
        
        Strategy:
        1. Product-level chunks: Aggregate all reviews per product
        2. Review-level chunks: Individual reviews for granular search
        
        Args:
            df: DataFrame with product and review data
            
        Returns:
            List of ReviewChunk objects
        """
        chunks = []
        
        # Handle multiple reviews per row (comma-separated)
        rows_data = []
        for _, row in df.iterrows():
            # Split comma-separated review fields
            user_names = str(row['user_name']).split(',')
            review_titles = str(row['review_title']).split(',')
            review_contents = str(row['review_content']).split(',')
            
            # Ensure all lists have same length
            max_len = max(len(user_names), len(review_titles), len(review_contents))
            user_names += ['Unknown'] * (max_len - len(user_names))
            review_titles += [''] * (max_len - len(review_titles))
            review_contents += [''] * (max_len - len(review_contents))
            
            for i in range(max_len):
                rows_data.append({
                    'product_id': row['product_id'],
                    'product_name': row['product_name'],
                    'category': row['category'],
                    'rating': row['rating'],
                    'user_name': user_names[i].strip(),
                    'review_title': review_titles[i].strip(),
                    'review_content': review_contents[i].strip(),
                    'about_product': row['about_product']
                })
        
        # Group by product
        product_groups = {}
        for data in rows_data:
            pid = data['product_id']
            if pid not in product_groups:
                product_groups[pid] = []
            product_groups[pid].append(data)
        
        # Create chunks
        for product_id, reviews in product_groups.items():
            first_review = reviews[0]
            
            # Strategy 1: Product Summary Chunk
            product_summary = self._create_product_summary(first_review, reviews)
            chunks.append(ReviewChunk(
                product_id=product_id,
                product_name=first_review['product_name'],
                category=first_review['category'],
                rating=first_review['rating'],
                review_title="Product Summary",
                review_content="",
                user_name="Aggregated",
                chunk_text=product_summary,
                chunk_type='product_summary'
            ))
            
            # Strategy 2: Individual Review Chunks
            for review in reviews:
                if review['review_content'] and len(review['review_content']) > 10:
                    review_text = self._create_review_chunk(review)
                    chunks.append(ReviewChunk(
                        product_id=product_id,
                        product_name=review['product_name'],
                        category=review['category'],
                        rating=review['rating'],
                        review_title=review['review_title'],
                        review_content=review['review_content'],
                        user_name=review['user_name'],
                        chunk_text=review_text,
                        chunk_type='individual_review'
                    ))
        
        print(f"Created {len(chunks)} chunks ({sum(1 for c in chunks if c.chunk_type == 'product_summary')} product summaries, "
              f"{sum(1 for c in chunks if c.chunk_type == 'individual_review')} individual reviews)")
        
        return chunks
    
    def _create_product_summary(self, product_info: Dict, reviews: List[Dict]) -> str:
        """Create a comprehensive product summary chunk"""
        name = product_info['product_name']
        category = product_info['category'].split('|')[-1]  # Last category
        rating = product_info['rating']
        about = product_info['about_product'][:500]  # Limit length
        
        # Aggregate review themes
        review_highlights = []
        for r in reviews[:5]:  # Top 5 reviews
            if r['review_title'] and r['review_content']:
                review_highlights.append(f"{r['review_title']}: {r['review_content'][:100]}")
        
        summary = f"""Product: {name}
Category: {category}
Overall Rating: {rating}/5
Key Features: {about}
Customer Feedback Highlights:
{chr(10).join(f"- {h}" for h in review_highlights)}"""
        
        return summary
    
    def _create_review_chunk(self, review: Dict) -> str:
        """Create an individual review chunk"""
        return f"""Product: {review['product_name']}
Review by {review['user_name']}
Title: {review['review_title']}
Rating Context: Product rated {review['rating']}/5
Review: {review['review_content']}"""
    
    def generate_embeddings(self, chunks: List[ReviewChunk]) -> np.ndarray:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of ReviewChunk objects
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_locally(self, chunks: List[ReviewChunk], embeddings: np.ndarray, 
                    save_dir: str = './rag_storage'):
        """
        Save chunks and embeddings locally for future retrieval
        
        Args:
            chunks: List of ReviewChunk objects
            embeddings: Numpy array of embeddings
            save_dir: Directory to save the data
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save chunks
        chunks_path = os.path.join(save_dir, 'chunks.pkl')
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save embeddings
        embeddings_path = os.path.join(save_dir, 'embeddings.npy')
        np.save(embeddings_path, embeddings)
        
        print(f"Saved {len(chunks)} chunks and embeddings to {save_dir}")
        print(f"Chunks file: {chunks_path} ({os.path.getsize(chunks_path) / 1024:.2f} KB)")
        print(f"Embeddings file: {embeddings_path} ({os.path.getsize(embeddings_path) / 1024:.2f} KB)")
    
    def load_locally(self, save_dir: str = './rag_storage'):
        """
        Load chunks and embeddings from local storage
        
        Args:
            save_dir: Directory containing saved data
        """
        chunks_path = os.path.join(save_dir, 'chunks.pkl')
        embeddings_path = os.path.join(save_dir, 'embeddings.npy')
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        self.embeddings = np.load(embeddings_path)
        
        print(f"Loaded {len(self.chunks)} chunks and embeddings from {save_dir}")
    
    def retrieve(self, query: str, product_filter: str = None, 
             top_k: int = 5, chunk_type: str = None) -> List[Tuple[ReviewChunk, float]]:
        """
        Retrieve most relevant chunks for a query with improved fuzzy product filtering
        
        Args:
            query: User query
            product_filter: Optional product name, brand or partial identifier
            top_k: Number of results to return
            chunk_type: Optional filter by chunk type ('product_summary' or 'individual_review')
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Generate query embedding for semantic search
        query_embedding = self.model.encode([query])[0]

        # Product filtering with fuzzy matching
        if product_filter:
            filter_str = product_filter.strip()
            if not filter_str:
                filtered_indices = list(range(len(self.chunks)))
            else:
                filter_lower = utils.default_process(filter_str)
                candidates = []
                
                for i, chunk in enumerate(self.chunks):
                    name_clean = utils.default_process(chunk.product_name)
                    
                    scores = [
                        fuzz.partial_ratio(filter_lower, name_clean),
                        fuzz.token_sort_ratio(filter_lower, name_clean),
                        fuzz.partial_token_sort_ratio(filter_lower, name_clean),
                    ]
                    best_score = max(scores)
                    
                    id_score = fuzz.partial_ratio(filter_lower, chunk.product_id.lower())
                    best_score = max(best_score, id_score * 0.9)
                    
                    if best_score >= 72:
                        candidates.append((i, best_score))
                
                if not candidates:
                    return []
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                filtered_indices = [idx for idx, _ in candidates[:150]]
        else:
            filtered_indices = list(range(len(self.chunks)))
        
        # Optional chunk type filter
        if chunk_type:
            filtered_indices = [
                i for i in filtered_indices 
                if self.chunks[i].chunk_type == chunk_type
            ]
        
        if not filtered_indices:
            return []
        
        # Semantic similarity on filtered candidates
        filtered_embeddings = self.embeddings[filtered_indices]
        
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.chunks[filtered_indices[i]], float(similarities[i]))
            for i in top_indices
        ]
        
        return results

    def search(self, query: str, product_filter: str = None, top_k: int = 3) -> str:
        """
        Search and format results for LLM consumption
        
        Args:
            query: User query
            product_filter: Optional product filter
            top_k: Number of results
            
        Returns:
            Formatted context string for LLM
        """
        results = self.retrieve(query, product_filter, top_k)
        
        if not results:
            return "No relevant reviews found."
        
        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(
                f"[Review {i}] (Relevance: {score:.2%})\n"
                f"Product: {chunk.product_name}\n"
                f"{chunk.chunk_text}\n"
            )
        
        return "\n---\n".join(context_parts)


if __name__ == "__main__":
    print("Product Review RAG System")
    print("This module is meant to be imported by the FastAPI app.")