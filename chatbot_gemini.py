from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time

os.environ["GOOGLE_API_KEY"] = "AIzaSyDCGJ8vFipG2NiX59Qgs6rInyptJYWJ2QA"
FAISS_DB_PATH = r'C:\Users\M.HARISH\PycharmProjects\PythonProject\faiss_index'

def load_faiss_db():
    try:
        EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS DB loaded successfully with E5 Large embeddings.")
        return db, embeddings
    except Exception as e:
        print(f"Error loading FAISS DB: {e}")
        exit()

vectorstore, embeddings = load_faiss_db()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

prompt_template = PromptTemplate(
    template="""
You are AI ChatBot, an expert assistant specializing in providing detailed and accurate responses strictly based on retrieved information. Your goal is to deliver factual, concise, and helpful answers without introducing speculative or fabricated content.

**INSTRUCTIONS:**
- Base all responses exclusively on the provided context. If the information is not available, clearly state that you do not have enough data to answer.
- Avoid generating information that is not explicitly stated or implied by the retrieved documents.
- Respond politely and informatively.
- Use headings, bullet points, and concise paragraphs for clarity and readability.
- Highlight key points, participants, and outcomes. Avoid over-explaining or speculating beyond the given data.
- Emphasize important actions, follow-ups, and next steps from meetings or discussions.

**ONGOING CONVERSATION:**
The following is a record of the conversation so far, including user queries and assistant responses. Use this to maintain context and provide answers in continuity with previous exchanges.

{chat_history}

**DOCUMENT CONTEXT (if available):**
The following context is retrieved from relevant documents related to the query.

{context}

**USER QUERY:**
{input}

**ASSISTANT RESPONSE:**
Provide a detailed response, keeping prior exchanges in mind. Refer to past questions and answers for continuity. Avoid repeating information unnecessarily but expand on new aspects related to the user's follow-up query.
""",
    input_variables=["context", "input", "chat_history"]
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "lambda_mult": 0.5  # Balances relevance (1) vs diversity (0)
    }
)

stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

history_aware_retriever = create_history_aware_retriever(
    prompt=prompt_template,
    retriever=retriever,
    llm=llm
)

qa_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=stuff_chain
)

def chatbot():
    print("AI Chatbot - Type 'exit' to stop")
    chat_history = []
    current_context = ""

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        formatted_chat_history = "\n".join(
            [f"User: {entry['question']}\nBot: {entry['answer']}" for entry in chat_history]
        )

        start = time.time()
        result = qa_chain.invoke({
            "input": query,
            "chat_history": formatted_chat_history,
            "context": current_context
        })
        end = time.time()

        response = result['answer']
        source_docs = result.get('source_documents', [])

        print(f"\n**Bot:** {response}")
        print("\n**Sources:**")
        for doc in source_docs:
            print(f"- {doc.metadata.get('source', 'Unknown source')}")
        print(f"**[Retrieved in {end - start:.2f} seconds]**\n")

        if source_docs:
            current_context += "\n".join([doc.page_content for doc in source_docs])

        chat_history.append({
            "question": query,
            "answer": response,
            "context": current_context
        })

        # Debug: Show evolving context (optional)
        # print(f"Updated Context: {current_context[:500]}...")  # Truncate for readability

if __name__ == "__main__":
    chatbot()
