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
    """Identify which SDGs are referenced in the module"""
    prompt = f"""
    [ROLE] UN SDG Assessment Expert
    [TASK] List the top 6 UN SUSTAINABLE GOALS numbers (e.g., '4', '13') that may be embedded in the teaching of this module, you can use your understanding of the 17 UN SDGs. You can be lenient and read between the lines. Return ONLY recognized SDG numbers separated by commas. If none, say 'None'.
    [MODULE DATA]
    Module Learning Objectives: {module_data[3]}
    Content: {module_data[2]}
    Teaching & Learning Methods: {module_data[2]}  # Update if different field
    Assessment: {module_data[4]}
    [INSTRUCTIONS] Return ONLY numbers separated by commas. If none, say 'None'.
    """
    response = pipe(prompt, max_new_tokens=50)[0]['generated_text']
    print("SDG References Response:", response) # Debugging output
    return re.findall(r'\d+', response)

def build_sdg_prompt(module_data, sdg_descriptions):
    """Build SDG-specific prompt with full descriptions"""
    context = "\n\n".join(sdg_descriptions)
    print("sdg descriptions:", context)  # Debugging output
    return f"""
    [ROLE] ESD Assessment Expert
    [INSTRUCTION] You have been provided with SDGs, and their descriptions, that are suspected to be embedded into a module taught at a university. you need to complete the tasks provided and respond as outlined in the task field and format field.
    [MODULE DATA]
    Module Learning Objectives: {module_data[3]}
    Content: {module_data[2]}
    Teaching & Learning Methods: {module_data[2]}  # Update if different field
    Assessment: {module_data[4]}
    
    [CONTEXT] {context}
    
    [TASK] For each Suggested SDG from CONTEXT answer the followinf:
    1. Is the SDG embedded in the module? If atleast 3 learning objectives out of the 15 for the Specific SDG are somewhat implied in the module content, the SDG is considered embedded. (Return The referenced SDG numbers separated by commas)
    2. How embedded are the referenced SDGs? (Rate 1-4)
    
    [FORMAT] JSON
    """

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
    """Generate structured prompts for each analysis section"""
    return f"""
    [ROLE] ESD Assessment Expert
    [INSTRUCTION] Answer strictly using module data and context. If no evidence exists, respond with "No evidence".
    [MODULE DATA]
    Module Learning Objectives: {module_data[3]}
    Content: {module_data[2]}
    Teaching & Learning Methods: {module_data[2]}  # Update if different field
    Assessment: {module_data[4]}
    
    [CONTEXT] {context}
    
    [TASK] Analyze for {section_name}:
    1. Direct references? (Yes/No)
    2. Embedding rating (0-4)
    3. Competency links (Explicit/Implicit/None) 
    [FORMAT] JSON
    """
    #redundant, q3

def build_synthesis_prompt(answers, context):
    """Final decision prompt with scoring thresholds"""
    Instructions=""" You need to determine if the module meets the ESD certification criteria based on the provided analyses. Criteria: if atleast one of (SDGs 8,9,10 or 12) AND atleast one of (SDGs 1,11,16,7,3,4,5,2) AND atleast one of (SDGs 13,14,15,6) AND atleast one Competency AND atleast one Pedagogical approach are embedded in the module, then the module is considered ESD certified. If not, it is not certified.
    """
    return f"""
    [ROLE] ESD Certification Auditor
    [INSTRUCTIONS] {Instructions}
    [ANALYSES] {answers}
    [CRITERIA] Requires ≥2 categories with score ≥3 
    [TASK] Final determination with RFC 2119-style justification
    [FORMAT] {{"decision": "Yes/No", "reason": "..."}}
    """
    #this needs to be changed, categories definition is not clear

# ======== Main Workflow ========
def main():
    # Initialize hardware
    login(token="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")
    torch.cuda.empty_cache()

    print("Available GPUs:", torch.cuda.device_count())  # Verify GPU count
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "SOLAR-10.7B-Instruct-v1.0")
    os.makedirs(model_path, exist_ok=True)

    # Configure memory allocation for all GPUs
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "10GB" for i in range(num_gpus)}
    max_memory["cpu"] = "60GB"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    if os.path.exists(os.path.join(model_path, "config.json")):
        print("Loading model from local storage...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_memory,
            offload_folder="./offload",
            torch_dtype=torch.float16,
            quantization_config=quant_config
        )
    else:
        print("Downloading model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            max_memory=max_memory,
            offload_folder="./offload",
            torch_dtype=torch.float16,
            quantization_config=quant_config
        )
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print("Model saved to:", model_path)

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
        max_new_tokens=400, #might need to increase this, competencies analysis results are being cut off
        temperature=0.0,  # Lower for reduced hallucination
        top_p=None,
        top_k=None,
        max_length=8192,
        repetition_penalty=1.1,
        return_full_text=False,
         do_sample=False 
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
