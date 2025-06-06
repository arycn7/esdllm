import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import torch
import pymupdf4llm

from helpers import dynamic_chunk_splitter, remove_section, extract_section, parse_and_extract
from langchain_community.llms import HuggingFacePipeline

# ======== Login ========
login(token="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")

import os

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes from codes/ to llm-env/
model_path = os.path.join(base_dir, "models", "llama_4_scout_model")

os.makedirs(model_path, exist_ok=True)

model_files_exist = os.path.exists(os.path.join(model_path, "config.json"))

if model_files_exist:
    print("🔁 Loading model from local storage...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
else:
    print("⬇️ Downloading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    print("✅ Model saved to:", model_path)

# ======== Extract Markdown from PDF ========
pdf_path = "3T. Core CE7T01.pdf"
md_text = pymupdf4llm.to_markdown(pdf_path)

# ======== Extract Module Data ========
modulecode, modulename, modulecontent, moduleLOs, moduleassesment = parse_and_extract(md_text)

# ======== Setup Pipeline ========
pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    top_k=3,
    temperature=0.4,
    return_full_text=False,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
)
llm = HuggingFacePipeline(pipeline=pipe)

# ======== Compose Prompt ========
details = f"""
Module Content: {modulecontent}
Module Learning Outcomes: {moduleLOs}
Module Assessment methods using format :Assessment Component,Assessment description, Learning Outcome(s)addressed, % of total, Assessment due date. {moduleassesment}
"""

sdgdesc = """ SDG 1: No Poverty - End poverty in all forms everywhere...
(Same SDG text as in original Colab, shortened here for brevity)
"""

query = "Which SDG's does this module definitely cover?"
prompt = f""" Instruction: Your task is to analyze a university module and respond to a user query - Base your answer strictly on the given module content and SDG descriptions. Keep your answer concise and do not use your own knowledge of SDGs.
Query: {query}
Module Details: {details}
Sustainable Development Goals description:{sdgdesc}
Answer:
"""

# ======== Generate Answer ========
output = llm.invoke(prompt)
print("Model Response:")
print(output)
