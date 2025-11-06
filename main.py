import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Check if the token is available
if not HF_TOKEN:
    print("❌ Missing HF_TOKEN in .env file!")
    exit()

# Initialize Hugging Face Inference client with your token
hf_client = InferenceClient(token=HF_TOKEN)

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
        # Limit input to first 3000 characters
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
def extract_keywords_text(text: str, max_keywords: int = 10) -> str:

    print("\n🔑 Extracting keywords")
    try:
        raw = hf_client.token_classification(
            text[:1500],
            model="ml6team/keyphrase-extraction-kbir-inspec",
        )
        # Normalize to dicts and handle both aggregated and raw BIO outputs
        items = []
        for r in raw:
            if isinstance(r, dict):
                items.append(r)
            else:
                items.append(dict(
                    start=getattr(r, "start", None),
                    end=getattr(r, "end", None),
                    label=getattr(r, "label", None),
                    entity_group=getattr(r, "entity_group", None),
                    score=getattr(r, "score", None),
                ))
        phrases = []
        aggregated = any(it.get("entity_group") for it in items)
        # Extract phrases based on aggregation
        if aggregated:
            for it in items:
                grp = str(it.get("entity_group", "")).upper()
                if "KEY" in grp and it.get("start") is not None and it.get("end") is not None:
                    phrases.append(text[it["start"]:it["end"]].strip())
        else:
            items.sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))
            cur_s = cur_e = None
            for s in items:
                lab = str(s.get("label", "")).upper()
                if lab.startswith("B"):
                    if cur_s is not None:
                        phrases.append(text[cur_s:cur_e])
                    cur_s, cur_e = s["start"], s["end"]
                elif lab.startswith("I") and cur_s is not None:
                    cur_e = s["end"]
                else:
                    if cur_s is not None:
                        phrases.append(text[cur_s:cur_e])
                    cur_s = cur_e = None
            if cur_s is not None:
                phrases.append(text[cur_s:cur_e])
        # Deduplicate and limit
        seen, out = set(), []
        for p in phrases:
            k = " ".join(p.split())
            if k and k not in seen:
                seen.add(k); out.append(k)

        max_keywords = max(1, min(max_keywords, 10))

        return ", ".join(out[:max_keywords]) if out else "No keyphrases found."
    except Exception as e:
        print(f"Warning: Keyword extraction error: {e}")
        return "Failed to extract keywords."

# Free-form text generation (decoder-only model; flat kwargs, no `parameters=`) <-- cause errors
def generate_text(prompt: str) -> str:
    print("\n📝 Generating text")
    try:
        response = hf_client.chat.completions.create(
            model="katanemo/Arch-Router-1.5B", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.5,
            top_p=0.9,
        )
        # Extract the generated text from the response
        generated_text = response.choices[0].message["content"].strip()
        return generated_text
    except Exception as e:
        print(f"Warning: Text generation error: {e}")
        return "Failed to generate text."


# Main execution
def main():
    print("=== AI Text Processor ===")
    file_path = input("Enter the name of the text file (e.g., input.txt): ").strip()
    text = load_text(file_path)

    summary = summarize_text(text)
    print("\n📘 Summary:\n", summary)

    print("\n")
    # Ask user how many keywords to extract (1-10)
    while True:
        try:
            num_keywords = int(input("How many keywords do you want to extract? (1-10): ").strip())
            if 1 <= num_keywords <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    keywords = extract_keywords_text(text, max_keywords=num_keywords)
    print("\n🏷️ Keywords:\n", keywords)

    print("\n")
    # Ask user for number of questions between 1 and 5
    while True:
        try:
            num_questions = int(input("How many questions do you want to generate? (1-5): ").strip())
            if 1 <= num_questions <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    prompt = (
        f"Generate {num_questions} multiple-choice questions based on the following text.\n"
        "Each question should have four answer options labeled A, B, C, and D.\n"
        "Do NOT repeat the answer options in a separate list.\n"
        "Format like this:\n"
        "Question 1: <question text>\n"
        "A) option 1\n"
        "B) option 2\n"
        "C) option 3\n"
        "D) option 4\n"
        "\n"
        f"Text:\n{text}"
    )


    gen = generate_text(prompt)
    print("\n🧪 Generation:\n", gen)


if __name__ == "__main__":
    main()

