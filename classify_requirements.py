import json
import os

import nltk # For sentence splitting (you might need to install it: pip install nltk)
import fitz # PyMuPDF for PDF handling
import re  # regular expressions
from nltk.tokenize import sent_tokenize

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Rule-based heuristics for requirement types
FUNCTIONAL_KEYWORDS = ["shall", "must", "should", "provide", "support"]
NON_FUNCTIONAL_KEYWORDS = ["performance", "security", "scalability", "availability"]
SYSTEM_KEYWORDS = ["database", "API", "server", "architecture"]
USER_KEYWORDS = ["user", "login", "profile", "role", "permissions"]
DEPENDENT_KEYWORDS = ["depends on", "prerequisite", "requires", "linked to"]

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

# Helper function for rule-based classification
def classify_requirement(sentences):
    requirements = []
    unclassified_requirements = []

    classification = None

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in FUNCTIONAL_KEYWORDS):
            classification = "Functional Requirement"
        elif any(keyword in sentence_lower for keyword in NON_FUNCTIONAL_KEYWORDS):
            classification = "Non-Functional Requirement"
        elif any(keyword in sentence_lower for keyword in SYSTEM_KEYWORDS):
            classification = "System Requirement"
        elif any(keyword in sentence_lower for keyword in USER_KEYWORDS):
            classification = "User Requirement"
        elif any(keyword in sentence_lower for keyword in DEPENDENT_KEYWORDS):
            classification = "Dependent Requirement"
        else:
            classification = "Unknown"
    
        if classification != "Unknown":
            requirements.append({"sentence": sentence, "category": classification})
        else:
            unclassified_requirements.append({"sentence": sentence, "category": classification})
    
    return requirements, unclassified_requirements

# Use LLaMA 3.2 (or OpenAI GPT-4) for refine classification
def classify_with_llm(unclassified_requirements):
    llm_classified = []

    # create the llm
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        format="json"
    )

    for requirement in unclassified_requirements:
        prompt_message = f"""
            Classify the following requirement into one of these categories:
            - Functional
            - Non-Functional
            - System
            - User
            - Dependent

            Requirement: "{requirement["sentence"]}"

            Respond with a JSON object:
            {{
                "sentence": "sentence",
                "classification": "Functional" (or other category)
            }}
        """

        try:
            # build the prompt using ChatPromptemplate
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant for requirement classification."),
                    ("human", "{query}"),
                ]
            )

            # use lcel chain for involking
            chain = prompt | llm

            # to see the response as it is being generated - you will have to use the stream option for invoking the llm
            response = chain.invoke({"query": prompt_message,})

            # return json.loads(response.content)
            llm_classified.append(response.content)
        except Exception as e:
            print(f"Error: {e}")
        
    return llm_classified

def merge_classifications(requirements, llm_classifications):
    
    """ # Merge rule-based and LLM classifications
    for i, req in enumerate(requirements):
        if llm_classifications and i < len(llm_classifications):
            req["llm_category"] = llm_classifications[i]["category"] """
    for requirement in llm_classifications:
        requirements.append(requirement)

    return requirements

def prepare_pdf_for_extraction(pdf_path):
    text = extract_text(pdf_path)
    text = clean_text(text)
    sentences = extract_sentences(text)
    # save_processed_data(sentences) # if you want to save the processed file uncomment this

    classified_requirements, unclassified_requirements = classify_requirement(sentences)

    # Use LLM to refine classifications - # Use LLaMA 3.2 for cases where keyword patterns are too rigid or to enhance accuracy.
    llm_classified_requirements = classify_with_llm(unclassified_requirements)

    final_requirements = merge_classifications(classified_requirements, llm_classified_requirements)

    print(final_requirements)

if __name__ == "__main__":
    pdf_file_path = "./docs/srs/SRS4.0.pdf" # Replace with your PDF file path
    prepare_pdf_for_extraction(pdf_path=pdf_file_path)