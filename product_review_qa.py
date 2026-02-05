"""
RAG + LLM Integration for Product Review Q&A
Uses local LLM or API for natural language responses
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

        # ── Early exit if no product context is available ─────────────
        if not product_filter and not self._has_product_mention(question):
            return (
                "I couldn't identify any specific product in your question.\n\n"
                "Please mention a product name, brand or model (e.g. 'Wayona cable', 'boAt Deuce', 'Ambrane fast charging') "
                "so I can find the relevant customer reviews."
            )
        
        else:
            # Retrieve relevant context
            context = self.rag.search(question, product_filter, top_k)
            
            if "No relevant reviews found" in context:
                return "I couldn't find any relevant reviews for that product or question."
            
            if use_llm:
                # Generate answer using LLM
                answer = self._generate_llm_answer(question, context, product_filter)
            else:
                # Return context directly
                answer = f"Based on the reviews:\n\n{context}"
            
            return answer
    
    def _generate_llm_answer(self, question: str, context: str, 
                            product_filter: str = None) -> str:
        """
        Generate natural language answer using LLM
        
        This is a template - you can integrate with:
        - OpenAI API (GPT-4, GPT-3.5)
        - Anthropic Claude API
        - Local models (Llama, Mistral via Ollama/HuggingFace)
        - Any other LLM API
        
        Args:
            question: User question
            context: Retrieved review context
            product_filter: Product name if filtered
            
        Returns:
            Natural language answer
        """
        prompt = self._create_prompt(question, context, product_filter)
        
        # Option 1: Use Anthropic Claude API (if available)
        # answer = self._call_anthropic_api(prompt)
        
        # Option 2: Use OpenAI API
        # answer = self._call_openai_api(prompt)
        
        # Option 3: Use local model (Ollama)
        # answer = self._call_ollama(prompt)
        
        # For demo: Simple rule-based response
        answer = self._generate_simple_answer(question, context, product_filter)
        
        return answer
    
    def _create_prompt(self, question: str, context: str, 
                      product_filter: str = None) -> str:
        """Create prompt for LLM"""
        product_mention = f" about {product_filter}" if product_filter else ""
        
        prompt = f"""You are a helpful product review analyst. Answer the user's question based ONLY on the provided customer reviews.

User Question: {question}{product_mention}

Customer Reviews:
{context}

Instructions:
- Provide a clear, concise answer based on the reviews
- Quote specific reviews when relevant
- If reviews have mixed opinions, mention both positive and negative aspects
- If the reviews don't contain enough information to answer, say so
- Be objective and balanced
- For comparison give proper understandable answer

Answer:"""
        
        return prompt
    
    def _generate_simple_answer(self, question: str, context: str, 
                               product_filter: str = None) -> str:
        """
        Generate a simple answer without LLM API
        This is a fallback/demo method - replace with actual LLM integration
        """
        # Extract key information from context
        lines = context.split('\n')
        review_texts = [line for line in lines if line.strip() and not line.startswith('[')]
        
        # Simple keyword-based analysis
        question_lower = question.lower()
        
        # Quality keywords
        if any(word in question_lower for word in ['quality', 'good', 'worth', 'reliable']):
            return self._analyze_quality(review_texts, product_filter)
        
        # Charging keywords
        elif any(word in question_lower for word in ['charging', 'charge', 'fast charging']):
            return self._analyze_charging(review_texts, product_filter)
        
        # Durability keywords
        elif any(word in question_lower for word in ['durable', 'lasting', 'break', 'sturdy']):
            return self._analyze_durability(review_texts, product_filter)
        
        # Camera keywords (for phones)
        elif any(word in question_lower for word in ['camera', 'photo', 'picture']):
            return self._analyze_camera(review_texts, product_filter)
        
        # General answer
        else:
            return self._generate_general_answer(review_texts, product_filter)
    
    def _analyze_quality(self, reviews: List[str], product: str = None) -> str:
        """Analyze product quality from reviews"""
        positive_keywords = ['good quality', 'excellent', 'great', 'premium', 'satisfied']
        negative_keywords = ['poor quality', 'bad', 'defective', 'issue', 'problem']
        
        positive_count = sum(1 for review in reviews if any(kw in review.lower() for kw in positive_keywords))
        negative_count = sum(1 for review in reviews if any(kw in review.lower() for kw in negative_keywords))
        
        product_name = f"the {product}" if product else "this product"
        
        if positive_count > negative_count:
            return f"Based on customer reviews, {product_name} has good quality. Customers mention positive aspects like build quality, durability, and value for money. Some reviews highlight: {self._extract_quote(reviews, positive_keywords)}"
        elif negative_count > positive_count:
            return f"Customer reviews indicate some quality concerns with {product_name}. {self._extract_quote(reviews, negative_keywords)}"
        else:
            return f"Customer opinions on {product_name}'s quality are mixed. Some praise its quality while others have experienced issues."
    
    def _analyze_charging(self, reviews: List[str], product: str = None) -> str:
        """Analyze charging performance"""
        fast_charging = sum(1 for review in reviews if 'fast charg' in review.lower())
        slow_charging = sum(1 for review in reviews if 'slow charg' in review.lower())
        
        product_name = f"the {product}" if product else "this cable"
        
        if fast_charging > 0:
            quote = next((r for r in reviews if 'fast charg' in r.lower()), '')
            return f"Yes, {product_name} supports fast charging according to customer reviews. {quote[:150]}..."
        elif slow_charging > 0:
            return f"Some customers report that {product_name} charges slower than expected."
        else:
            return f"Customers mention charging functionality but specific details about charging speed are limited in the reviews."
    
    def _analyze_durability(self, reviews: List[str], product: str = None) -> str:
        """Analyze product durability"""
        durable_keywords = ['durable', 'sturdy', 'strong', 'unbreakable', 'lasting']
        fragile_keywords = ['broke', 'broken', 'tear', 'fragile']
        
        durable = sum(1 for review in reviews if any(kw in review.lower() for kw in durable_keywords))
        fragile = sum(1 for review in reviews if any(kw in review.lower() for kw in fragile_keywords))
        
        product_name = f"The {product}" if product else "This product"
        
        if durable > fragile:
            return f"{product_name} appears to be durable based on customer feedback. Reviews mention its sturdy construction and ability to withstand regular use."
        elif fragile > durable:
            return f"Some customers report durability issues with {product_name}. There are mentions of the product breaking or wearing out."
        else:
            return f"Durability feedback for {product_name} is limited in the available reviews."
    
    def _analyze_camera(self, reviews: List[str], product: str = None) -> str:
        """Analyze camera quality (for phones/cameras)"""
        good_camera = sum(1 for review in reviews if any(kw in review.lower() for kw in ['good camera', 'great camera', 'excellent camera', 'camera quality']))
        bad_camera = sum(1 for review in reviews if any(kw in review.lower() for kw in ['poor camera', 'bad camera', 'camera issue']))
        
        product_name = product or "this device"
        
        if good_camera > 0:
            return f"According to reviews, {product_name} has good camera quality. Customers are satisfied with the photo quality."
        elif bad_camera > 0:
            return f"Some reviews mention camera quality issues with {product_name}."
        else:
            return f"I couldn't find specific information about camera quality for {product_name} in the available reviews."
    
    def _generate_general_answer(self, reviews: List[str], product: str = None) -> str:
        """Generate general answer from reviews"""
        product_name = product or "the product"
        sample_reviews = '\n'.join(reviews[:3])
        return f"Based on customer reviews for {product_name}:\n\n{sample_reviews}\n\nCustomers generally share their experiences with quality, performance, and value."
    
    def _extract_quote(self, reviews: List[str], keywords: List[str]) -> str:
        """Extract a relevant quote from reviews"""
        for review in reviews:
            if any(kw in review.lower() for kw in keywords):
                # Extract sentence containing keyword
                sentences = review.split('.')
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in keywords):
                        return f'"{sentence.strip()}."'
        return ""


# Example integration with actual LLM APIs
class LLMIntegrations:
    """Example integrations with various LLM APIs"""
    
    @staticmethod
    def call_openai(prompt: str, api_key: str) -> str:
        """
        Example: OpenAI GPT integration
        
        Install: pip install openai
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful product review analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except ImportError:
            return "OpenAI library not installed. Run: pip install openai"
    
    @staticmethod
    # def call_anthropic(prompt: str, api_key: str) -> str:
    #     """
    #     Example: Anthropic Claude integration
        
    #     Install: pip install anthropic
    #     """
    #     try:
    #         from anthropic import Anthropic
    #         client = Anthropic(api_key=api_key)
            
    #         response = client.messages.create(
    #             model="claude-3-haiku-20240307",
    #             max_tokens=300,
    #             messages=[
    #                 {"role": "user", "content": prompt}
    #             ]
    #         )
            
    #         return response.content[0].text
    #     except ImportError:
    #         return "Anthropic library not installed. Run: pip install anthropic"
    

    @staticmethod
    def call_llm_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Call Groq API with strong preference for clean JSON output
        """
        try:
            from groq import Groq
            
            # api_key = os.getenv("api_key")
            if not GROQ_API_KEY:
                return "Error: GROQ_API_KEY environment variable not set. Get one at https://console.groq.com/keys"

            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,                # very important for structured output
                max_tokens=512,
                top_p=0.95,
                response_format={"type": "json_object"},  # ← forces JSON mode on supported models
                timeout=30
            )

            content = response.choices[0].message.content.strip()
            return content

        except ImportError:
            return "Error: 'groq' library not installed. Run: pip install groq"
        except Exception as e:
            return f"Groq API error: {str(e)}"


def demo_qa_system():
    """Demonstrate the complete Q&A system"""
    
    print("=" * 80)
    print("PRODUCT REVIEW Q&A SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize RAG system
    rag = ProductReviewRAG()
    
    # Load pre-generated embeddings if available
    if os.path.exists('./rag_storage'):
        print("\nLoading existing embeddings...")
        rag.load_locally()
    else:
        print("\nGenerating new embeddings...")
        # You would load your data here
        print("Note: Run product_review_rag.py first to generate embeddings")
        return
    
    # Initialize Q&A system
    qa = ProductReviewQA(rag)
    
    # Demo questions
    questions = [
        ("Is the Wayona cable durable?", "Wayona Nylon Braided USB to Lightning Fast Charging and Data Sync Cable Compatible for iPhone 13"),
        ("Does the Ambrane cable support fast charging?", "Ambrane Unbreakable 60W / 3A Fast Charging 1.5m Braided Type C Cable for Smartphones"),
        ("What's the quality like?", None),
        ("How long does it last?", "boAt"),
    ]
    
    print("\n" + "=" * 80)
    print("QUESTION ANSWERING EXAMPLES")
    print("=" * 80)
    
    for question, product_filter in questions:
        print(f"\n{'=' * 80}")
        print(f"Question: {question}")
        if product_filter:
            print(f"Product: {product_filter}")
        print("-" * 80)
        
        answer = qa.answer_question(question, product_filter, top_k=2)
        print(f"\nAnswer:\n{answer}")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nYou can now ask questions about the products!")
    print("Examples:")
    print("  - 'Is the cable durable?' with product 'Wayona'")
    print("  - 'Does it charge fast?' with product 'Ambrane'")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            product = input("Product filter (or press Enter for all): ").strip() or None
            
            if question:
                answer = qa.answer_question(question, product, top_k=3)
                print(f"\n{answer}\n")
        except KeyboardInterrupt:
            break
    
    print("\nThank you for using the Product Review Q&A System!")


if __name__ == "__main__":
    demo_qa_system()