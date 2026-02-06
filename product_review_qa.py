"""
RAG + LLM Integration for Product Review Q&A
Uses local LLM or API for natural language responses with improved formatting
"""

from product_review_rag import ProductReviewRAG, ReviewChunk
from typing import List, Tuple
import os
from dotenv import load_dotenv

# Try to load local .env (won't hurt in production)
load_dotenv()

GROQ_API_KEY = os.getenv("api_key")


class ProductReviewQA:
    """Question Answering system combining RAG with LLM"""
    
    def __init__(self, rag_system: ProductReviewRAG = None):
        """
        Initialize QA system
        
        Args:
            rag_system: Optional pre-initialized RAG system
        """
        self.rag = rag_system or ProductReviewRAG()
    
    def answer_question(self, question: str, product_filter: str = None, 
                       top_k: int = 3, use_llm: bool = True) -> str:
        """
        Answer a question about product reviews
        
        Args:
            question: User's question
            product_filter: Optional product name/ID filter
            top_k: Number of reviews to retrieve
            use_llm: Whether to use LLM for answer generation
            
        Returns:
            Natural language answer
        """
        # Retrieve relevant context
        context = self.rag.search(question, product_filter, top_k)
        
        if "No relevant reviews found" in context:
            return "I couldn't find any relevant reviews to answer your question. Please try rephrasing or ask about a different aspect of the product."
        
        if use_llm:
            # Generate answer using LLM
            answer = self._generate_llm_answer(question, context, product_filter)
        else:
            # Return context directly
            answer = f"Based on the reviews:\n\n{context}"
        
        return answer
    
    def _generate_llm_answer(self, question: str, context: str, 
                            product_filter: str = None) -> str:
        """Generate natural language answer using LLM (Groq)"""
        prompt = self._create_prompt(question, context, product_filter)
        
        # Use Groq API for answer generation
        llm = LLMIntegrations()
        answer = llm.call_llm_groq(
            prompt=prompt,
            model="llama-3.3-70b-versatile"
        )
        
        return answer
    
    def _create_prompt(self, question: str, context: str, 
                      product_filter: str = None) -> str:
        """Create prompt for LLM with enhanced formatting"""
        product_mention = f" about {product_filter}" if product_filter else ""
        
        prompt = f"""You are a friendly and helpful product review assistant. Your goal is to provide clear, easy-to-understand answers based on real customer reviews.

User Question: {question}{product_mention}

Customer Reviews:
{context}

Instructions for your answer:
1. Start with a direct answer to the question
2. Use simple, conversational language that anyone can understand
3. Structure your response with clear paragraphs:
   - First paragraph: Direct answer summary
   - Second paragraph: Positive points from reviews (if any)
   - Third paragraph: Concerns or negative points (if any)
   - Final paragraph: Balanced conclusion
4. When quoting reviews, use phrases like "One customer mentioned..." or "According to reviews..."
5. Use bullet points (•) only when listing 3+ similar points
6. Avoid technical jargon unless the question specifically asks for it
7. If reviews have mixed opinions, be balanced and explain both sides
8. For comparison questions, create a clear contrast between products
9. End with a helpful recommendation or summary when appropriate
10. Keep the total response concise (2-4 paragraphs maximum)
11. keep it brief 

Remember: Your answer should feel like a knowledgeable friend explaining the product, not a formal report.

Answer:"""
        
        return prompt


# LLM Integration Class
class LLMIntegrations:
    """LLM API integrations"""
    
    @staticmethod
    def call_llm_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Call Groq API for natural language generation
        """
        try:
            from groq import Groq
            
            if not GROQ_API_KEY:
                return "⚠️ API key not configured. Please set the 'api_key' environment variable."

            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful product review assistant. Provide clear, conversational answers based on customer reviews."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.9,
                timeout=30
            )

            content = response.choices[0].message.content.strip()
            return content

        except ImportError:
            return "❌ Error: 'groq' library not installed. Run: pip install groq"
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"


if __name__ == "__main__":
    print("Product Review QA System - Enhanced Edition")
    print("This module is meant to be imported by the FastAPI app.")