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
    prompt = f"""You are analyzing a university module to find the MOST RELEVANT UN SDGs.

MODULE DETAILS:
Title: {module_data[0]}
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

UN SDGs (pick the 5 most relevant to the best of your knowledge and to this module):
1-No Poverty, 2-Zero Hunger, 3-Good Health, 4-Quality Education, 5-Gender Equality, 
6-Clean Water, 7-Affordable Energy, 8-Decent Work, 9-Industry Innovation, 
10-Reduced Inequalities, 11-Sustainable Cities, 12-Responsible Consumption, 
13-Climate Action, 14-Life Below Water, 15-Life on Land, 16-Peace Justice, 17-Partnerships

Think about:
- What field is this module in?
- What skills does it teach?
- What topics does it cover?
- How could it impact society?

OUTPUT: Just 5 numbers separated by commas (most relevant first)
EXAMPLE: 4, 9, 8, 17, 13

ANSWER:"""

    response = pipe(prompt, max_new_tokens=50)[0]['generated_text']
    print("SDG References Response:", response)
    numbers = re.findall(r'\b([1-9]|1[0-7])\b', response)
    return numbers[:5]

 
def build_sdg_prompt(module_data, sdg_descriptions):
    # Build individual SDG entries
    sdg_entries = []
    for desc in sdg_descriptions:
        sdg_match = re.search(r'SDG (\d+):', desc)
        if sdg_match:
            sdg_num = sdg_match.group(1)
            sdg_name = desc.split(':')[1].split('\n')[0].strip()
            sdg_entries.append(f"SDG {sdg_num}: {sdg_name}")
    
    return f"""Analyze each SDG below for this university module:

MODULE:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}

ANALYZE THESE SDGs:
{chr(10).join(sdg_entries)}

For each SDG listed above, determine if it's embedded in the module.

EXAMPLE OF OUTPUT FORMAT - JSON array with ALL SDGs listed above:
[
  {{"SDG_NUMBER": 4, "SDG_NAME": "Quality Education", "EMBEDDED": "Yes"}},
  {{"SDG_NUMBER": 8, "SDG_NAME": "Decent Work", "EMBEDDED": "No"}},
  {{"SDG_NUMBER": 13, "SDG_NAME": "Climate Action", "EMBEDDED": "Yes"}},
  {{"SDG_NUMBER": 9, "SDG_NAME": "Industry Innovation", "EMBEDDED": "No"}},
  {{"SDG_NUMBER": 17, "SDG_NAME": "Partnerships", "EMBEDDED": "Yes"}}
]

JSON OUTPUT:"""



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
    if section_name == "Competencies":
        # Extract individual competencies from context
        competency_list = []
        for ctx in context:
            competencies = re.findall(r'([A-Z][^:]*competency): ([^.]*\.)', ctx)
            competency_list.extend(competencies)
        
        competency_text = "\n".join([f"- {name}: {desc}" for name, desc in competency_list[:6]])
        
        return f"""Analyze this university module for ESD competencies:

MODULE:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

ESD COMPETENCIES TO CHECK:
{competency_text}

TASK: For each competency above, determine if the module develops it.

OUTPUT FORMAT:
{{
    "competencies_found": [
        {{"name": "Systems thinking competency", "present": "Yes", "evidence": "Module covers complex systems"}},
        {{"name": "Critical thinking competency", "present": "No", "evidence": "No evidence found"}}
    ],
    "overall_rating": 3
}}

JSON OUTPUT:"""
    
    else:  # Pedagogy
        return f"""Analyze this university module for ESD pedagogical approaches:

MODULE:
Learning Objectives: {module_data[3]}
Content: {module_data[2]}
Assessment: {module_data[4]}

PEDAGOGICAL APPROACHES TO CHECK:
{chr(10).join(context)}

TASK: Identify which pedagogical approaches are used in this module.

OUTPUT FORMAT:
{{
    "approaches_found": [
        {{"name": "Learner-centred approach", "present": "Yes", "evidence": "Students construct knowledge"}},
        {{"name": "Action-oriented learning", "present": "No", "evidence": "No practical projects"}}
    ],
    "overall_assessment": "Module uses some ESD pedagogical approaches"
}}

JSON OUTPUT:"""


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

def extract_valid_json(response_text):
    """Extract valid JSON from model response"""
    # Try to find JSON patterns
    patterns = [
        r'\[[\s\S]*?\]',  # Array
        r'\{[\s\S]*?\}'   # Object
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            try:
                # Test if it's valid JSON
                import json
                json.loads(match)
                return match
            except:
                continue
    
    # If no valid JSON found, return the response
    return response_text.strip()


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
        max_new_tokens=100,
        temperature=0.1,        # Very low but not 0 - allows minimal variance
        top_p=0.7,             # More constrained than 0.9
        do_sample=True,        # Enable sampling but constrained
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        repetition_penalty=1.1
                      
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
        cleaned_result = extract_valid_json(sdg_response[0]['generated_text'])

        results["SDG"] = cleaned_result
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
        cleaned_response = extract_valid_json(response[0]['generated_text'])
        results[section] = cleaned_response
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
