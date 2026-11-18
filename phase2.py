print("SANJEEVANI PHASE2 STARTED")

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient

# Apni Tavily API Key yahan daalein
os.environ["TAVILY_API_KEY"] = "tvly-rfS5J-9aNUryk4GAvefZJo1Xkb1w2BRxGgvJrONBp1YFeLqw"
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

print("1. Loading Embeddings...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("2. Loading FAISS Database...")
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

print("3. Loading Ollama Model...")
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048
)

CUSTOM_PROMPT_TEMPLATE = """You are an expert medical assistant.

Answer the question strictly using ONLY the provided context.

CRITICAL RULE:
First, check if the EXACT medicine, disease, or topic asked in the Question is actually present in the Context. 
- Do NOT confuse similar-sounding names (for example, do not answer about "Senna" if the user asked about "Senarest").
- If the exact subject of the question is NOT clearly mentioned in the context, output ONLY this exact phrase and nothing else:
"I don't have enough information in the provided medical sources to answer this."

If the exact subject IS in the context, write a detailed, well-structured response of around 10 to 15 lines in simple English:
1. Overview Paragraph: Start directly with a clear 4-5 line paragraph explaining the main concept and definition from the context.
2. Key Points: Follow the paragraph with 5 to 7 detailed bullet points covering uses, causes, mechanisms, or important facts mentioned in the context.

Context:
{context}

Question:
{question}

Detailed Answer:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

print("4. Creating QA Chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("\nSANJEEVANI IS READY!")

user_query = input("\nWrite Query Here: ")

# Step 1: Pehle PDF Database (FAISS) check karega
response = qa_chain.invoke({"query": user_query})
final_answer = response["result"]
source_type = "LOCAL MEDICAL PDFs"

# Step 2: Agar PDF mein exact answer nahi mila, toh Tavily Web Search chalega
missing_phrases = [
    "i don't have enough information",
    "not mentioned in the provided",
    "no specific information provided"
]

if any(phrase in final_answer.lower() for phrase in missing_phrases):
    print("\n[Info] Exact answer not in PDFs. Searching Web via Tavily...")
    source_type = "TAVILY WEB SEARCH"
    
    search_results = tavily.search(query=user_query, max_results=3)
    web_context = "\n\n".join([res["content"] for res in search_results["results"]])
    
    web_prompt = prompt.format(context=web_context, question=user_query)
    final_answer = llm.invoke(web_prompt).content

print(f"\nRESULT (Source: {source_type}):")
print(final_answer)

if source_type == "LOCAL MEDICAL PDFs":
    print("\nSOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        print(doc.metadata)
