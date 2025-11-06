import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import inspect

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Check if the token is available
if not HF_TOKEN:
    print("❌ Missing HF_TOKEN in .env file!")
    exit()

# Initialize Hugging Face Inference client with your token
hf_client = InferenceClient(token=HF_TOKEN)

# Function to check what parameters the client methods accept
def print_signature(client_method, name):
    signature = inspect.signature(client_method)
    print(f"\nFunction signature for hf_client.{name}():\n{signature}")
    print("\nParameters details:")
    for name, param in signature.parameters.items():
        print(f"  - {name}: {param.kind}, default: {param.default}")

# Call these functions to see the arguments
print_signature(hf_client.summarization, "summarization")
print_signature(hf_client.text_generation, "text_generation")

# Load text from a file
def load_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        exit()

# Summarization function
def summarize_text(text: str) -> str:
    print("\n🧾 Summarizing text...")
    try:
        input_text = text[:3000]
        response = hf_client.summarization(
            input_text,
            model="facebook/bart-large-cnn",
        )
        # Handle both list and single object responses
        if isinstance(response, list):
            summary = response[0].summary_text.strip()
        else:
            summary = response.summary_text.strip()
        return summary
    except Exception as e:
        print(f"Warning: Summarization error: {e}")
        return "Failed to summarize."

# Keyword extraction function
def extract_keywords_text(text: str) -> str:
    """
    Extracts keywords using a fine-tuned T5 model via text generation.
    """
    print("\n🔑 Extracting keywords...")
    prompt = f"generate keywords: {text[:2000]}"
    
    try:
        response = hf_client.text_generation(
            prompt,
            model="Voicelab/vlt5-base-keywords",
            max_new_tokens=50,  
            temperature=0.1,    
            do_sample=False, 
        )
        keywords = response.strip()
        return keywords
    except Exception as e:
        print(f"Warning: Keyword extraction error: {e}")
        return "Failed to extract keywords."

# Main execution
def main():
    print("=== AI Text Processor ===")
    file_path = input("Enter the name of the text file (e.g., input.txt): ").strip()
    text = load_text(file_path)

    summary = summarize_text(text)
    print("\n📘 Summary:\n", summary)

    keywords = extract_keywords_text(text)
    print("\n🏷️ Keywords:\n", keywords)

if __name__ == "__main__":
    main()

