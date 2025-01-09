import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time
import speech_recognition as sr
import pyttsx3

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

# Function to handle real-time audio transcription
def transcribe_realtime_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("🎤 Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Error with the speech recognition service."

# Function to convert text to speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
def chatbot():
    st.title("AI Chatbot Interface")
    st.write("Interact with the chatbot by typing or speaking your queries below.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'context' not in st.session_state:
        st.session_state.context = ""

    # Text input
    user_query = st.text_input("💬 Type your query:", "")

    # Real-time audio input
    if st.button("🎙️"):
        transcribed_query = transcribe_realtime_audio()
        st.write(f"Transcribed Query: {transcribed_query}")
        user_query = transcribed_query

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

        st.write("### 🤖 Bot:")
        st.write(response)

        if st.button("🔊"):
            speak_text(response)

        st.write(f"**Retrieved in {end - start:.2f} seconds**")

if __name__ == "__main__":
    chatbot()
