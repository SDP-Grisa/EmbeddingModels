"""
FastAPI E-commerce App for Product Review RAG System
Clean architecture with category browsing, product listings, and review chatbot
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from typing import Optional, List, Dict
import json

from product_review_rag import ProductReviewRAG
from product_review_qa import ProductReviewQA, LLMIntegrations

# ────────────────────────────────────────────────────────────────
# Initialize FastAPI App
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="E-commerce Product Review System")

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ────────────────────────────────────────────────────────────────
# Load RAG System (Global)
# ────────────────────────────────────────────────────────────────
rag_system = None
qa_system = None

def load_rag():
    """Load RAG system on startup"""
    global rag_system, qa_system
    storage_dir = "./rag_storage"
    
    if not os.path.exists(storage_dir):
        print(f"⚠️  Warning: RAG storage not found at {storage_dir}")
        print("Please run the indexing script first to generate embeddings.")
        return False
    
    try:
        rag_system = ProductReviewRAG()
        rag_system.load_locally(save_dir=storage_dir)
        qa_system = ProductReviewQA(rag_system)
        print(f"✅ Loaded {len(rag_system.chunks)} review chunks successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load RAG system: {e}")
        return False

# ────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────
def get_all_categories() -> List[Dict]:
    """Extract all unique categories from chunks"""
    if not rag_system:
        return []
    
    categories = {}
    for chunk in rag_system.chunks:
        # Parse category hierarchy (e.g., "Electronics|Accessories|Cables")
        category_parts = chunk.category.split('|')
        main_category = category_parts[0] if category_parts else "Other"
        
        if main_category not in categories:
            categories[main_category] = {
                'name': main_category,
                'product_count': 0,
                'products': set()
            }
        
        categories[main_category]['products'].add(chunk.product_id)
    
    # Convert to list and count products
    result = []
    for cat_name, cat_data in categories.items():
        result.append({
            'name': cat_name,
            'product_count': len(cat_data['products']),
            'slug': cat_name.lower().replace(' ', '-').replace('&', 'and')
        })
    
    return sorted(result, key=lambda x: x['product_count'], reverse=True)

def get_products_by_category(category_slug: str) -> List[Dict]:
    """Get all unique products in a category"""
    if not rag_system:
        return []
    
    category_name = category_slug.replace('-', ' ').replace('and', '&').title()
    products = {}
    
    for chunk in rag_system.chunks:
        category_parts = chunk.category.split('|')
        main_category = category_parts[0] if category_parts else "Other"
        
        if main_category.lower() == category_name.lower():
            if chunk.product_id not in products:
                products[chunk.product_id] = {
                    'id': chunk.product_id,
                    'name': chunk.product_name,
                    'rating': chunk.rating,
                    'category': chunk.category,
                    'reviews': []
                }
            
            # Collect reviews
            if chunk.chunk_type == 'individual_review':
                products[chunk.product_id]['reviews'].append({
                    'title': chunk.review_title,
                    'content': chunk.review_content[:200] + '...' if len(chunk.review_content) > 200 else chunk.review_content,
                    'user': chunk.user_name
                })
    
    return list(products.values())

def get_product_details(product_id: str) -> Optional[Dict]:
    """Get complete product details including all reviews"""
    if not rag_system:
        return None
    
    product_data = {
        'id': product_id,
        'name': '',
        'category': '',
        'rating': 0,
        'about': '',
        'reviews': [],
        'summary': ''
    }
    
    for chunk in rag_system.chunks:
        if chunk.product_id == product_id:
            # Basic info
            if not product_data['name']:
                product_data['name'] = chunk.product_name
                product_data['category'] = chunk.category
                product_data['rating'] = chunk.rating
            
            # Product summary
            if chunk.chunk_type == 'product_summary':
                product_data['summary'] = chunk.chunk_text
                # Extract about section
                if 'Key Features:' in chunk.chunk_text:
                    about_section = chunk.chunk_text.split('Key Features:')[1].split('Customer Feedback')[0]
                    product_data['about'] = about_section.strip()
            
            # Individual reviews
            elif chunk.chunk_type == 'individual_review':
                product_data['reviews'].append({
                    'title': chunk.review_title,
                    'content': chunk.review_content,
                    'user': chunk.user_name
                })
    
    return product_data if product_data['name'] else None

# ────────────────────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    product_id: str
    question: str

# ────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load RAG system on startup"""
    load_rag()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - Display all categories"""
    categories = get_all_categories()
    return templates.TemplateResponse("home.html", {
        "request": request,
        "categories": categories
    })

@app.get("/category/{category_slug}", response_class=HTMLResponse)
async def category_page(request: Request, category_slug: str):
    """Category page - Display all products in category"""
    products = get_products_by_category(category_slug)
    category_name = category_slug.replace('-', ' ').replace('and', '&').title()
    
    return templates.TemplateResponse("category.html", {
        "request": request,
        "category_name": category_name,
        "category_slug": category_slug,
        "products": products
    })

@app.get("/product/{product_id}", response_class=HTMLResponse)
async def product_page(request: Request, product_id: str):
    """Product detail page with chatbot"""
    product = get_product_details(product_id)
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return templates.TemplateResponse("product.html", {
        "request": request,
        "product": product
    })

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint for product-specific questions"""
    if not qa_system:
        return JSONResponse({
            "error": "RAG system not initialized"
        }, status_code=503)
    
    try:
        # Get product details for context
        product = get_product_details(request.product_id)
        if not product:
            return JSONResponse({
                "error": "Product not found"
            }, status_code=404)
        
        # Generate answer using QA system
        answer = qa_system.answer_question(
            question=request.question,
            product_filter=product['name'],
            top_k=4,
            use_llm=True
        )
        
        return JSONResponse({
            "answer": answer,
            "product_name": product['name']
        })
        
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to generate answer: {str(e)}"
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_loaded": rag_system is not None,
        "chunks_count": len(rag_system.chunks) if rag_system else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)