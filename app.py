import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from gtts import gTTS  # Google Text-to-Speech library for generating audio
import tempfile
import time

# Load Google API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets["google"]["api_key"]

FAISS_DB_PATH = 'faiss_index'

@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_llm(api_key):
    """
    Load the ChatGoogleGenerativeAI LLM with the provided API key.
    """
    llm_instance = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        api_key=api_key
    )
    return llm_instance

vectorstore, embeddings = load_faiss_db()
llm = load_llm(GOOGLE_API_KEY)

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
        "lambda_mult": 0.5
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

# Streamlit UI
def chatbot():
    st.title("AI Chatbot Interface")
    st.write("Interact with the chatbot by typing your queries below.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None

    user_query = st.text_input("You:", "")

    if user_query:
        start = time.time()
        result = qa_chain.invoke({
            "input": user_query,
            "chat_history": "",
            "context": st.session_state.context
        })
        end = time.time()

        response = result['answer']

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": response
        })

        st.write("### Bot:")
        st.write(response)

        # Generate TTS audio using gTTS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts = gTTS(text=response, lang='en')
            tts.save(temp_audio.name)
            st.session_state.audio_path = temp_audio.name  # Save the file path for later playback

        st.write(f"**Retrieved in {end - start:.2f} seconds**")

    # Add the speaker emoji button
    if st.session_state.audio_path and st.button("🔊"):
        with open(st.session_state.audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    chatbot()
