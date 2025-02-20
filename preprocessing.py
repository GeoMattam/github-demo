import json
import os

import nltk # For sentence splitting (you might need to install it: pip install nltk)
import fitz # PyMuPDF for PDF handling
import re  # regular expressions
from nltk.tokenize import sent_tokenize

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Define requirement classification patterns
patterns = {
    "Functional": [
        r"\b(the system shall|must be able to|should allow|is required to)\b",
        r"\b(input|output|process|store|validate|display)\b"
    ],
    "Non_Functional": [
        r"\b(performance|scalability|security|availability|reliability|usability)\b",
        r"\b(response time|throughput|latency|uptime|encryption|data privacy)\b"
    ],
    "System": [
        r"\b(hardware|software|API|infrastructure|platform|configuration)\b",
        r"\b(OS|database|memory|network|integration|protocol)\b"
    ],
    "User": [
        r"\b(user role|user type|actor|end-user|customer|administrator)\b",
        r"\b(as a|user must be able to|user should|persona)\b"
    ],
    "Dependent": [
        r"\b(depends on|requires|prerequisite|based on|linked to|dependent on)\b",
        r"\b(reference to requirement|requires completion of)\b"
    ]
}

# Extract text from pdf
def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        # extract text from PDF
        text = ""
        for page in doc:
            text += page.get_text("text")
        
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def clean_text(text):
    # Clean the Extracted Text
    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)
    
    # Remove headers, footers, page numbers (common patterns)
    text = re.sub(r"Page\s\d+ of \d+", "", text)  # Example: "Page 1 of 10"
    text = re.sub(r"\b(Confidential|Draft|Version\s\d+)\b", "", text, flags=re.I)  # Metadata
    
    text = text.strip()

    # Normalize Whitespace and Encoding (important!)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")  # Handle encoding issues
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with single space, remove leading/trailing spaces

    return text

def extract_sentences(text):
    # Split into Sentences/Paragraphs (using NLTK)
    sentences = sent_tokenize(text) # Split into sentences

    return sentences

def save_processed_data(sentences):
    with open("preprocessed_srs.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=4)

def classify_requirements(sentences, patterns):
    classified_reqs = {
        "Functional": [],
        "Non_Functional": [],
        "System": [],
        "User": [],
        "Dependent": []
    }

    for sentence in sentences:
        classified = False
        for category, rules in patterns.items():
            if any(re.search(rule, sentence, re.IGNORECASE) for rule in rules):
                classified_reqs[category].append(sentence)
                classified = True
                break
        
        # If no match, store it under "Uncategorized"
        if not classified:
            classified_reqs.setdefault("Uncategorized", []).append(sentence)
    
    return classified_reqs

def classify_with_llm(sentence):
    # create the llm
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    prompt_message = f"""
        Classify the following requirement into one of these categories:
        - Functional
        - Non-Functional
        - System
        - User
        - Dependent

        Requirement: "{sentence}"

        Respond with a JSON object:
        {{
            "category": "Functional" (or other category),
            "reason": "Brief explanation"
        }}
    """

    # build the prompt using ChatPromptemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{query}"),
        ]
    )

    # use lcel chain for involking
    chain = prompt | llm

    # to see the response as it is being generated - you will have to use the stream option for invoking the llm
    response = chain.invoke(
        {
            "query": prompt_message,
        }
    )

    return response.content

def prepare_pdf_for_extraction(pdf_path):
    text = extract_text(pdf_path)
    text = clean_text(text)
    sentences = extract_sentences(text)
    # save_processed_data(sentences) # if you want to save the processed file uncomment this

    classified_requirements = classify_requirements(sentences, patterns)
    # Use LLaMA 3.2 for cases where keyword patterns are too rigid or to enhance accuracy.
    for sentence in classified_requirements.get("Uncategorized", []):
        result = classify_with_llm(sentence)
        print(result)  # Check the LLM output for debugging

if __name__ == "__main__":
    pdf_file_path = "./docs/srs/SRS4.0.pdf" # Replace with your PDF file path
    prepare_pdf_for_extraction(pdf_path=pdf_file_path)