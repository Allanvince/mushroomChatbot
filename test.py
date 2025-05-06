from transformers import pipeline
import fitz  # PyMuPDF
from typing import Optional, Tuple
import os

class MushroomChatbot:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the mushroom farming chatbot
        
        Args:
            model_name: Name of the QA model to use (default: roberta-base-squad2)
        """
        self.model = None
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """Initialize the QA pipeline"""
        try:
            self.model = pipeline(
                "question-answering",
                model=self.model_name
            )
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        try:
            doc = fitz.open(pdf_path)
            return " ".join(page.get_text() for page in doc).strip()
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        pdf_path: Optional[str] = None,
        max_context_length: int = 5000
    ) -> Tuple[str, float]:
        """
        Answer a question based on provided context or PDF
        
        Args:
            question: The question to answer
            context: Direct text context (optional)
            pdf_path: Path to PDF file (optional)
            max_context_length: Maximum context length in characters
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not self.model:
            raise RuntimeError("Model not initialized")

        # Get context from either PDF or direct text
        if pdf_path:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            context = self.extract_text_from_pdf(pdf_path)
        elif not context:
            raise ValueError("Either context or pdf_path must be provided")

        # Limit context size
        context = context[:max_context_length]
        
        result = self.model(question=question, context=context)
        
        return result['answer'], round(result['score'], 4)

    def interactive_session(self, pdf_path: str):
        """
        Start an interactive Q&A session (CLI mode)
        
        Args:
            pdf_path: Path to the PDF knowledge base
        """
        try:
            context = self.extract_text_from_pdf(pdf_path)
            print(f"\nExtracted {len(context.split())} words from PDF")
            print("Type 'exit' or 'quit' to end the session\n")
            
            while True:
                question = input("Enter your question about mushroom farming: ").strip()
                
                if question.lower() in ['exit', 'quit']:
                    print("\nEnding Q&A session. Goodbye!")
                    break
                
                if not question:
                    print("Please enter a valid question.\n")
                    continue
                
                try:
                    print("\nSearching for answer...")
                    answer, confidence = self.answer_question(question, context=context)
                    print(f"\nAnswer: {answer}")
                    print(f"Confidence: {confidence:.2f} (higher is better)")
                except Exception as e:
                    print(f"\nError finding answer: {e}")
                
                print("\n" + "="*50 + "\n")
                
        except Exception as e:
            print(f"Failed to start interactive session: {e}")

if __name__ == "__main__":
    # CLI Mode (when run directly)
    PDF_PATH = "./data/mushroom_farming_guide.pdf"  # Default path
    
    print("Loading Mushroom Farming Chatbot...")
    try:
        bot = MushroomChatbot()
        print("\nStarting Interactive Q&A System")
        bot.interactive_session(PDF_PATH)
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")