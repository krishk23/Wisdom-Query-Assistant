import time
import os
import json
import random
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorize_documents import embeddings
from deep_translator import GoogleTranslator
from googlesearch import search

# Set up working directory and API configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
os.environ["GROQ_API_KEY"] = config_data["GROQ_API_KEY"]

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

def chat_chain(vectorstore):
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    return chain

def fetch_daily_quote():
    query = "Bhagavad Gita inspirational quotes"
    results = list(search(query, num_results=5))  # Convert generator to list
    if results:
        return random.choice(results)
    return "Explore the Bhagavad Gita and Yoga Sutras for timeless wisdom!"

# Streamlit UI
st.set_page_config(
    page_title="Bhagavad Gita & Yoga Sutras Assistant",
    page_icon="🕉️",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">Wisdom Query Assistant</h1>
        <p style="font-size: 18px;">Explore timeless wisdom with the guidance of a knowledgeable assistant.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User name functionality
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if not st.session_state.chat_started:
    st.markdown("<h3 style='text-align: center;'>Welcome! Before we begin, please enter your name:</h3>", unsafe_allow_html=True)
    user_name = st.text_input("Enter your name:", placeholder="Your Name", key="name_input")
    start_button = st.button("Start Chat")

    if start_button and user_name.strip():
        st.session_state.user_name = user_name.strip()
        st.session_state.chat_started = True
        st.success(f"Hello {st.session_state.user_name}! How can I assist you today?")

# Display the daily quote
quote = fetch_daily_quote()
st.markdown(
    f"""
    <div style="text-align: center; background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h4>🌟 Daily Wisdom: <a href="{quote}" target="_blank">{quote}</a></h4>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.chat_started:
    # Set up vectorstore and chat chain
    vectorstore = setup_vectorstore()
    chain = chat_chain(vectorstore)

    # Select language
    selected_language = st.selectbox("Select your preferred language:", options=[
        "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Malayalam", "Kannada",
        "Punjabi", "Odia", "Maithili", "Sanskrit", "Santali", "Kashmiri", "Nepali", "Dogri", "Manipuri", "Bodo",
        "Sindhi", "Assamese", "Konkani", "Awadhi", "Rajasthani", "Haryanvi", "Bihari", "Chhattisgarhi", "Magahi"
    ], index=0)

    # Display chat history
    st.markdown("### 💬 Chat History")
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history:
            st.markdown(f"**{st.session_state.user_name}:** {chat['question']}")
            st.markdown(f"**Assistant:** {chat['answer']}")
            st.markdown("---")

    # Input box for new query
    st.markdown(f"### Ask a new question, {st.session_state.user_name}:")
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_input("Your question:", key="query_input", placeholder="Type your query here...")
        submitted = st.form_submit_button("Submit")

    if submitted and user_query.strip():
        start_time = time.time()
        response = chain({"question": user_query.strip()})
        end_time = time.time()

        answer = response.get("answer", "No answer found.")
        source_documents = response.get("source_documents", [])
        execution_time = round(end_time - start_time, 2)

        # Translate response if needed
        if selected_language != "English":
            translator = GoogleTranslator(source="en", target=selected_language.lower())
            translated_answer = translator.translate(answer)
        else:
            translated_answer = answer

        # Save chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "question": user_query.strip(),
            "answer": translated_answer
        })

        # Display source documents if available
        if source_documents:
            with st.expander("📜 Source Documents"):
                for i, doc in enumerate(source_documents):
                    st.write(f"**Document {i + 1}:** {doc.page_content}")

        st.write(f"**🌟 Enlightened Response:** {translated_answer}")
        st.write(f"_Response time: {execution_time} seconds_")

    # Sharing options
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://wa.me/?text=Explore%20the%20Bhagavad%20Gita%20%26%20Yoga%20Sutras%20Assistant!%20Check%20it%20out%20here:%20https://your-platform-link" target="_blank">
                <img src="https://img.icons8.com/color/48/whatsapp.png" alt="WhatsApp" style="margin-right: 10px;">
            </a>
            <a href="https://www.linkedin.com/shareArticle?mini=true&url=https://your-platform-link&title=Explore%20Wisdom%20with%20Our%20Assistant" target="_blank">
                <img src="https://img.icons8.com/color/48/linkedin.png" alt="LinkedIn">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )












# import time
# import os
# import json
# import random
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from vectorize_documents import embeddings
# from deep_translator import GoogleTranslator  # For multilingual support

# # Set up working directory and API configuration
# working_dir = os.path.dirname(os.path.abspath(__file__))
# config_data = json.load(open(f"{working_dir}/config.json"))
# os.environ["GROQ_API_KEY"] = config_data["GROQ_API_KEY"]

# def setup_vectorstore():
#     persist_directory = f"{working_dir}/vector_db_dir"
#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embeddings
#     )
#     return vectorstore

# def chat_chain(vectorstore):
#     from langchain_groq import ChatGroq  # Import the LLM class

#     llm = ChatGroq(
#         model="llama-3.1-70b-versatile",  # Replace with your LLM of choice
#         temperature=0  # Set low temperature to reduce hallucinations
#     )
#     retriever = vectorstore.as_retriever()  # Retrieve relevant chunks
#     memory = ConversationBufferMemory(
#         llm=llm,
#         output_key="answer",
#         memory_key="chat_history",
#         return_messages=True
#     )

#     # Build the conversational retrieval chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",  # Define how documents are combined
#         memory=memory,
#         verbose=True,
#         return_source_documents=True
#     )
#     return chain

# # Streamlit UI
# st.set_page_config(
#     page_title="Bhagavad Gita & Yoga Sutras Assistant",
#     page_icon="🕉️",  # Custom meaningful favicon
#     layout="wide"
# )

# # Title and description with enhanced styling
# st.markdown(
#     """
#     <div style="text-align: center;">
#         <h1 style="color: #4CAF50;">Wisdom Query Assistant</h1>
#         <p style="font-size: 18px;">Explore timeless wisdom with the guidance of a knowledgeable assistant.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # Daily Wisdom Quote
# daily_quotes = [
#     "You have the right to work, but never to the fruit of work. – Bhagavad Gita",
#     "Yoga is the journey of the self, through the self, to the self. – Bhagavad Gita",
#     "When meditation is mastered, the mind is unwavering like the flame of a lamp in a windless place. – Bhagavad Gita",
#     "Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment. – Buddha",
# ]
# st.markdown(
#     f"""
#     <div style="text-align: center; background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
#         <h4>🌟 Daily Wisdom: {random.choice(daily_quotes)}</h4>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # Theme Toggle
# theme = st.radio("Choose a Theme:", options=["Light", "Dark"], index=0, horizontal=True)
# if theme == "Dark":
#     st.markdown(
#         """
#         <style>
#         body { background-color: #121212; color: white; }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# vectorstore = setup_vectorstore()
# chain = chat_chain(vectorstore)

# # Initialize session state
# if "user_name" not in st.session_state:
#     st.session_state.user_name = ""

# if "chat_started" not in st.session_state:
#     st.session_state.chat_started = False

# # Language options
# languages = [
#     "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Malayalam", "Kannada",
#     "Punjabi", "Odia", "Maithili", "Sanskrit", "Santali", "Kashmiri", "Nepali", "Dogri", "Manipuri", "Bodo",
#     "Sindhi", "Assamese", "Konkani", "Awadhi", "Rajasthani", "Haryanvi", "Bihari", "Chhattisgarhi", "Magahi"
# ]

# # Input for user name
# if not st.session_state.chat_started:
#     st.markdown("<h3 style='text-align: center;'>Welcome! Before we begin, please enter your name:</h3>", unsafe_allow_html=True)
#     user_name = st.text_input("Enter your name:", placeholder="Your Name", key="name_input")
#     start_button = st.button("Start Chat")

#     if start_button and user_name.strip():
#         st.session_state.user_name = user_name.strip()
#         st.session_state.chat_started = True
#         st.success(f"Hello {st.session_state.user_name}! How can I assist you today?")

# # Chat functionality
# if st.session_state.chat_started:
#     st.markdown(f"<h3 style='text-align: center;'>Hello {st.session_state.user_name}! Ask me anything:</h3>", unsafe_allow_html=True)

#     # Language selection dropdown
#     selected_language = st.selectbox("Select your preferred language:", options=languages, index=0)

#     # User input and buttons
#     user_query = st.text_input("💬 Type your question:", placeholder="Type your query here...", key="query_box")
#     submit_button = st.button("Submit")

#     if submit_button and user_query.strip():
#         # Generate response
#         start_time = time.time()
#         response = chain({"question": user_query.strip()})
#         end_time = time.time()

#         answer = response.get("answer", "No answer found.")
#         source_documents = response.get("source_documents", [])
#         execution_time = round(end_time - start_time, 2)

#         # Translate response
#         if selected_language != "English":
#             translator = GoogleTranslator(source="en", target=selected_language.lower())
#             translated_answer = translator.translate(answer)
#         else:
#             translated_answer = answer

#         # Display answer
#         st.markdown("---")
#         st.markdown(f"### 🌟 Enlightened Response:")
#         st.write(translated_answer)

#         # Display source documents
#         if source_documents:
#             st.markdown("### 📜 Source Documents:")
#             for i, doc in enumerate(source_documents):
#                 with st.expander(f"Source Document {i + 1}"):
#                     st.write(doc.page_content)
#         else:
#             st.markdown("No source documents available.")

#         # Execution time
#         st.markdown(f"<p style='font-size: 14px;'>Response Time: <strong>{execution_time}</strong> seconds</p>", unsafe_allow_html=True)

#     # Sharing options with icons
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style="text-align: center;">
#             <a href="https://wa.me/?text=Explore%20the%20Bhagavad%20Gita%20%26%20Yoga%20Sutras%20Assistant!%20Check%20it%20out%20here:%20https://your-platform-link" target="_blank">
#                 <img src="https://img.icons8.com/color/48/whatsapp.png" alt="WhatsApp" style="margin-right: 10px;">
#             </a>
#             <a href="https://www.linkedin.com/shareArticle?mini=true&url=https://your-platform-link&title=Explore%20Wisdom%20with%20Our%20Assistant" target="_blank">
#                 <img src="https://img.icons8.com/color/48/linkedin.png" alt="LinkedIn">
#             </a>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
















# import time
# import os
# import json
# import streamlit as st
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from vectorize_documents import embeddings  # Import embeddings from the vectorization script
# from deep_translator import GoogleTranslator  # Import Google Translator for multilingual support

# # Set up working directory and API configuration
# working_dir = os.path.dirname(os.path.abspath(__file__))
# config_data = json.load(open(f"{working_dir}/config.json"))
# os.environ["GROQ_API_KEY"] = config_data["GROQ_API_KEY"]

# def setup_vectorstore():
#     persist_directory = f"{working_dir}/vector_db_dir"
#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embeddings
#     )
#     return vectorstore

# def chat_chain(vectorstore):
#     from langchain_groq import ChatGroq  # Import the LLM class

#     llm = ChatGroq(
#         model="llama-3.1-70b-versatile",  # Replace with your LLM of choice
#         temperature=0  # Set low temperature to reduce hallucinations
#     )
#     retriever = vectorstore.as_retriever()  # Retrieve relevant chunks
#     memory = ConversationBufferMemory(
#         llm=llm,
#         output_key="answer",
#         memory_key="chat_history",
#         return_messages=True
#     )

#     # Build the conversational retrieval chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",  # Define how documents are combined
#         memory=memory,
#         verbose=True,
#         return_source_documents=True
#     )
#     return chain

# # Streamlit UI
# st.set_page_config(page_title="Bhagavad Gita & Yoga Sutras Assistant", layout="wide")

# # Title and description with enhanced styling
# st.markdown(
#     """
#     <div style="text-align: center;">
#         <h1 style="color: #4CAF50;">Wisdom Query Assistant</h1>
#         <p style="font-size: 18px;">Explore timeless wisdom with the guidance of a knowledgeable assistant.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# vectorstore = setup_vectorstore()
# chain = chat_chain(vectorstore)

# # Initialize session state for user name and chat
# if "user_name" not in st.session_state:
#     st.session_state.user_name = ""

# if "chat_started" not in st.session_state:
#     st.session_state.chat_started = False

# # Language options
# languages = [
#     "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Malayalam", "Kannada",
#     "Punjabi", "Odia", "Maithili", "Sanskrit", "Santali", "Kashmiri", "Nepali", "Dogri", "Manipuri", "Bodo",
#     "Sindhi", "Assamese", "Konkani", "Awadhi", "Rajasthani", "Haryanvi", "Bihari", "Chhattisgarhi", "Magahi"
# ]

# # Input for user name
# if not st.session_state.chat_started:
#     st.markdown("<h3 style='text-align: center;'>Welcome! Before we begin, please enter your name:</h3>", unsafe_allow_html=True)
#     user_name = st.text_input("Enter your name:", placeholder="Your Name", key="name_input")
#     start_button = st.button("Start Chat")

#     if start_button and user_name.strip():
#         st.session_state.user_name = user_name.strip()
#         st.session_state.chat_started = True
#         st.success(f"Hello {st.session_state.user_name}! How can I assist you today?")

# # Chat functionality
# if st.session_state.chat_started:
#     st.markdown(f"<h3 style='text-align: center;'>Hello {st.session_state.user_name}! Ask me about Wisdom:</h3>", unsafe_allow_html=True)

#     # Language selection dropdown
#     selected_language = st.selectbox("Select your preferred language:", options=languages, index=0)

#     # User input and submit button at the bottom
#     user_query = st.text_input("💬 Your question:", placeholder="Type your query here...", key="query_box")
#     submit_button = st.button("Submit")

#     if submit_button and user_query.strip():
#         # Generate response
#         start_time = time.time()
#         response = chain({"question": user_query.strip()})
#         end_time = time.time()

#         answer = response.get("answer", "No answer found.")
#         source_documents = response.get("source_documents", [])
#         execution_time = round(end_time - start_time, 2)

#         # Translate the answer based on selected language
#         if selected_language != "English":
#             translator = GoogleTranslator(source="en", target=selected_language.lower())
#             translated_answer = translator.translate(answer)
#         else:
#             translated_answer = answer

#         # Display the answer
#         st.markdown("---")
#         st.markdown(f"### 🌟 Enlightened Response:")
#         st.write(translated_answer)

#         # Display source documents
#         if source_documents:
#             st.markdown("### 📜 Source Documents:")
#             for i, doc in enumerate(source_documents):
#                 with st.expander(f"Source Document {i + 1}"):
#                     st.write(doc.page_content)
#         else:
#             st.markdown("No source documents available.")

#         # Display execution time
#         st.markdown(f"<p style='font-size: 14px;'>Response Time: <strong>{execution_time}</strong> seconds</p>", unsafe_allow_html=True)














