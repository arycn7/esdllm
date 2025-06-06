import os
import torch  # moved up to do empty_cache early
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
import pymupdf4llm

from helpers import dynamic_chunk_splitter, remove_section, extract_section, parse_and_extract
from langchain_community.llms import HuggingFacePipeline

# ======== Login ========
login(token="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")

# Clear CUDA cache early to free GPU memory
torch.cuda.empty_cache()

print("")

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
        max_memory={
            0: "10GB",  # adjust if needed to slightly below total GPU VRAM
            "cpu": "60GB"
        },
        offload_folder="./offload",  # folder for offloaded tensors
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
        max_memory={
            0: "10GB",
            "cpu": "60GB"
        },
        offload_folder="./offload",
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
pipe = pipeline(
    "text-generation",
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

sdgdesc=""" SDG 1: No Poverty - End poverty in all forms everywhere, including extreme poverty, through social protection systems, access to basic services, and economic inclusion.

SDG 2: Zero Hunger - End hunger, achieve food security and improved nutrition, and promote sustainable agriculture by supporting small-scale farmers and equitable food distribution.

SDG 3: Good Health and Well-being - Ensure healthy lives and promote well-being for all ages by improving healthcare access, reducing disease, and addressing mental health.

SDG 4: Quality Education - Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all, focusing on marginalized groups.

SDG 5: Gender Equality - Achieve gender equality and empower all women and girls by ending discrimination, violence, and harmful practices, and ensuring equal opportunities.

SDG 6: Clean Water and Sanitation - Ensure availability and sustainable management of water and sanitation for all, addressing scarcity and improving hygiene.

SDG 7: Affordable and Clean Energy - Ensure access to affordable, reliable, sustainable, and modern energy for all through renewable sources and energy efficiency.

SDG 8: Decent Work and Economic Growth - Promote sustained, inclusive economic growth, full employment, and decent work with safe environments and fair labor rights.

SDG 9: Industry, Innovation, and Infrastructure - Build resilient infrastructure, promote inclusive industrialization, and foster innovation, especially in developing regions.

SDG 10: Reduced Inequalities - Reduce inequality within and among countries through policies for economic inclusion and representation of marginalized groups.

SDG 11: Sustainable Cities and Communities - Make cities inclusive, safe, resilient, and sustainable via improved urban planning, housing, and green public spaces.

SDG 12: Responsible Consumption and Production - Ensure sustainable consumption and production patterns through resource efficiency, waste reduction, and corporate accountability.

SDG 13: Climate Action - Combat climate change and its impacts through urgent action, education, and integration into national policies.

SDG 14: Life Below Water - Conserve and sustainably use oceans, seas, and marine resources by reducing pollution and overfishing.

SDG 15: Life on Land - Protect terrestrial ecosystems, halt biodiversity loss, combat desertification, and promote sustainable forest management.

SDG 16: Peace, Justice, and Strong Institutions - Promote peaceful societies, provide access to justice, and build accountable institutions to reduce corruption and violence.

SDG 17: Partnerships for the Goals - Strengthen global partnerships for sustainable development through finance, technology, trade, and multistakeholder cooperation."""


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
