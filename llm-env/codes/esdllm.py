import os
import torch
import faiss
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=50)

from sentence_transformers import SentenceTransformer
from helpers import parse_and_extract, dynamic_chunk_splitter  # Ensure these exist
import pymupdf4llm  # Ensure this is installed

# ======== Configuration ========
MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CONTEXT_DOC_PATH = "esd_context.txt"  # Your large context document
PDF_PATH = "3T. Core CE7T01.pdf"
pedagogy_doc_path = "pedagogy.txt"  # Path to pedagogy document
competencies_doc_path = "competencies.txt"  # Path to competencies document
# ======== Helper Functions ========
def smart_chunk_sdg_descriptions(context_text):
    """Split context document into individual SDG chunks"""
    pattern = r'(SDG \d+:.*?)(?=SDG \d+:|$)'
    matches = re.findall(pattern, context_text, flags=re.DOTALL)
    sdg_chunks = {}
    for match in matches:
        sdg_num = re.search(r'SDG (\d+):', match)
        if sdg_num:
            sdg_chunks[f'SDG {sdg_num.group(1)}'] = match.strip()
    return sdg_chunks

def get_referenced_sdgs(module_data, pipe):
    prompt = f"""You are an SDG classification expert. Your task is to identify UN Sustainable Development Goal numbers that relate to this university module.

MODULE CONTENT:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

TASK: Return exactly 5 SDG numbers (1-17) that best relate to this module, in order of relevance.

RESPONSE FORMAT: Only numbers separated by commas
EXAMPLE: 4, 8, 13, 9, 17

RESPONSE:"""

    response = pipe(prompt, max_new_tokens=30)[0]['generated_text']
    print("SDG References Response:", response)
    # Extract only the numbers from the response
    numbers = re.findall(r'\b([1-9]|1[0-7])\b', response.split('\n')[0])
    return numbers[:5]
 
 
def build_sdg_prompt(module_data, sdg_descriptions):
    return f"""You are an SDG assessment expert. Analyze if this module embeds the listed SDGs.

MODULE DATA:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

SDG DESCRIPTIONS:
{chr(10).join(sdg_descriptions)}

ANALYSIS TASK:
For each SDG above, determine:
1. Is it embedded in the module? (Yes/No)
2. If Yes, rate strength (1-4: 1=weak, 4=very strong)

REQUIRED OUTPUT FORMAT:
Respond with ONLY valid JSON array. No other text.

[
  {{
    "SDG_NUMBER": 4,
    "SDG_NAME": "Quality Education",
    "EMBEDDED": "Yes",
    "RATING": 3
  }},
  {{
    "SDG_NUMBER": 13,
    "SDG_NAME": "Climate Action",
    "EMBEDDED": "No", 
    "RATING": ""
  }}
]

JSON RESPONSE:"""


def initialize_embedder():
    """Initialize sentence transformer embedder"""
    return SentenceTransformer(EMBEDDING_MODEL)

def create_faiss_index(chunks, embedder):
    """Create FAISS index from document chunks"""
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_context(query, embedder, index, chunks, k=4):
    """Retrieve relevant context chunks"""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def build_section_prompt(section_name, module_data, context):
    context_text = "\n".join(context)
    
    return f"""You are an ESD assessment expert analyzing a university module for {section_name}.

MODULE DATA:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

REFERENCE FRAMEWORK:
{context_text}

ANALYSIS QUESTIONS:
1. Are there direct references to {section_name.lower()} concepts?
2. How strongly embedded are {section_name.lower()} elements?
3. What type of competency links exist?

REQUIRED JSON OUTPUT:
{{
    "analysis": "Brief analysis in 1-2 sentences",
    "direct_reference": true,
    "embedding_rating": 3,
    "competency_links": "Explicit"
}}

RULES:
- direct_reference: true or false only
- embedding_rating: number 0-4 only
- competency_links: "Explicit", "Implicit", or "None" only
- analysis: maximum 50 words

JSON RESPONSE:"""


def build_synthesis_prompt(answers, context):
    return f"""You are an ESD certification auditor making the final decision.

ANALYSIS RESULTS:
{str(answers)}

CERTIFICATION CRITERIA:
- Must have at least one SDG from economic category (8,9,10,12) 
- Must have at least one SDG from social category (1,2,3,4,5,11,16)
- Must have at least one SDG from environmental category (6,7,13,14,15)
- Must show competency development evidence
- Must show pedagogical approach evidence

DECISION TASK:
Based on the analyses above, does this module meet ESD certification criteria?

REQUIRED JSON OUTPUT:
{{
    "decision": "Yes",
    "reason": "Module meets criteria with embedded SDGs and competency evidence"
}}

RULES:
- decision: "Yes" or "No" only
- reason: maximum 20 words explaining decision

JSON RESPONSE:"""


# ======== Main Workflow ========
def main():
    torch.cuda.empty_cache()
    print("Available GPUs:", torch.cuda.device_count())

    # Model path (already stored locally)
    model_path = "/home/support/llm/Llama-3.1-8B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model in full FP16, no quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},           # Fully load on GPU 0
        torch_dtype=torch.float16,    # Use FP16
        offload_folder=None
    )
    print("Model loaded successfully.")
    # Print device allocation
    print("Model device map:", model.hf_device_map)

    # Initialize RAG components
    embedder = initialize_embedder()
    with open(CONTEXT_DOC_PATH) as f:
        context_text = f.read()
    with open(pedagogy_doc_path) as f:
        pedagogy_text = f.read()
    with open(competencies_doc_path) as f:
        competencies_text = f.read()

    pedagogy_chunks = text_splitter.split_text(pedagogy_text)
    competencies_chunks = text_splitter.split_text(competencies_text)
    pedagogy_index = create_faiss_index(pedagogy_chunks, embedder)
    competencies_index = create_faiss_index(competencies_chunks, embedder)
    
    md_text = pymupdf4llm.to_markdown(PDF_PATH)
    module_data = parse_and_extract(md_text)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,     # Increased for complete responses
        temperature=0.0,        # Set to 0 for consistent outputs
        top_p=0.9,             # Focused sampling
        do_sample=False,       # Deterministic output
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        repetition_penalty=1.05
    )


    # Initialize results
    results = {}

    # === SDG Analysis ===
    sdg_chunks = smart_chunk_sdg_descriptions(context_text)
    referenced_sdgs = get_referenced_sdgs(module_data, pipe)
    sdg_descriptions = [sdg_chunks[f'SDG {num}'] for num in referenced_sdgs if f'SDG {num}' in sdg_chunks]

    if sdg_descriptions:
        sdg_prompt = build_sdg_prompt(module_data, sdg_descriptions)
        sdg_response = pipe(sdg_prompt)
        results["SDG"] = sdg_response[0]['generated_text']
    else:
        results["SDG"] = '{"sdg_coverage": "No evidence"}'
    print("SDG Analysis Result:")
    print(results["SDG"])
    # === Other Sections (Competencies, Pedagogy) ===
    analysis_sections = {
        "Competencies": "ESD competency framework",
        "Pedagogy": "ESD pedagogical approaches"
    }
    for section, query in analysis_sections.items():
        if section == "Competencies":
            context_chunks = competencies_chunks
            faiss_index = competencies_index    
        else:
            context_chunks = pedagogy_chunks
            faiss_index = pedagogy_index
        context = retrieve_context(query, embedder, faiss_index, context_chunks)
        print(f"Retrieved context for {section}: {context}")
         # Build and run the prompt for each section
        prompt = build_section_prompt(section, module_data, context)
        response = pipe(prompt)
        results[section] = response[0]['generated_text']
        print(f"{section} Analysis Result:")
        print(results[section])

    # === Final Synthesis ===
    synthesis_context = retrieve_context("ESD certification criteria", embedder, faiss_index, context_chunks) #needs to be changed
    final_prompt = build_synthesis_prompt(results, synthesis_context) #synthesis context needs to be changed
    print("Final Prompt for Synthesis:")
    print(final_prompt)
    print("___")
    final_decision = pipe(final_prompt)
    
    print("Final Certification Decision:")
    print(final_decision[0]['generated_text'])

if __name__ == "__main__":
    main()
