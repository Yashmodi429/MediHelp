
pip install langchain==0.1.12
pip install langchain-google-genai==0.0.7
pip install langchain-community==0.0.29
pip install streamlit==1.32.2
pip install pyngrok==7.1.5
pip install google-generativeai>=0.3.2
pip install streamlit streamlit_webrtc googletrans==4.0.0-rc1 langchain langchain-google-genai
pip install SpeechRecognition pyaudio

import os
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import subprocess
from pyngrok import ngrok

# Set up Google API Key (ensure that you have the key available)
os.environ['GOOGLE_API_KEY'] = userdata.get('GEMINI_API_KEY')

# Streamlit page config
st.set_page_config(page_title="AI Medical Assistant 🤖", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body { background-color: white; font-family: 'Arial', sans-serif; color: #333; }
        .header-title { font-size: 40px; text-align: center; color: #D32F2F; margin-top: 30px; font-weight: bold; }
        .header-subtitle { font-size: 18px; text-align: center; color: #555; margin-top: 10px; }
        .chat-box { background-color: #fff; border-radius: 15px; padding: 20px; box-shadow: 0 0 15px rgba(0, 0, 0, 0.1); height: 400px; overflow-y: scroll; }
        .footer { text-align: center; padding: 15px; background-color: #D32F2F; color: white; font-size: 16px; position: fixed; width: 100%; bottom: 0; font-weight: bold; }
        .user-message { background-color: #DCF8C6; padding: 12px; border-radius: 10px; font-size: 16px; color: #333; }
        .ai-message { background-color: #EAEAEA; padding: 12px; border-radius: 10px; font-size: 16px; color: #333; }
        .highlighted-term { background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 5px; cursor: pointer; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">🤖 Welcome to Your AI Medical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Developed with care by Yash Samir Modi</div>', unsafe_allow_html=True)

# Gemini setup
gemini = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-001',
    temperature=0.1,
    convert_system_message_to_human=True
)

# Custom prompt for medical conditions
SYS_PROMPT = """
You are a medical assistant designed to provide clear, accurate, and up-to-date information about various medical conditions.

When given a medical condition, respond with the following format:

Condition Name: {{Condition Name}}
Overview: {{Brief summary of the condition}}

Common Symptoms:
{{List of common symptoms}}

Causes:
{{List of common causes or risk factors}}

Diagnosis:
{{How it is usually diagnosed}}

Treatment Options:
{{Medications, therapies, or procedures}}

Prevention Tips:
{{How to prevent it, if possible}}

When to See a Doctor:
{{Red flags or emergency signs}}

Sources:
{{Trusted source references like Mayo Clinic, WebMD, WHO}}

If the user asks about the specific aspects of the condition like "Diagnosis," "Symptoms," or "Treatment," provide relevant information for those subcategories.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm_chain = prompt | gemini
streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Chat History Box
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="chat-content">', unsafe_allow_html=True)
for msg in streamlit_msg_history.messages:
    with st.chat_message(msg.type):
        if msg.type == "human":
            st.markdown(f'<div class="user-message">{msg.content}</div>', unsafe_allow_html=True)
        else:
            response = msg.content
            terms_to_highlight = ["Diabetes", "Insulin", "Symptoms", "Treatment", "Diagnosis", "Hypertension", "Cancer", "Asthma"]
            for term in terms_to_highlight:
                response = response.replace(term, f'<span class="highlighted-term">{term}</span>')
            st.markdown(f'<div class="ai-message">{response}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat Input
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
user_prompt = st.chat_input("Type your medical condition here...")
if user_prompt:
    st.chat_message("human").markdown(user_prompt)
    with st.chat_message("ai"):
        try:
            config = {"configurable": {"session_id": "any"}}
            response = conversation_chain.invoke({"input": user_prompt}, config)
            if hasattr(response, "content"):
                response_content = response.content
                terms_to_highlight = ["Diabetes", "Insulin", "Symptoms", "Treatment", "Diagnosis"]
                for term in terms_to_highlight:
                    response_content = response_content.replace(term, f'<span class="highlighted-term">{term}</span>')
                st.markdown(f'<div class="ai-message">{response_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">{str(response)}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed by Yash Samir Modi | Medical Assistant Chatbot | Powered by AI</div>', unsafe_allow_html=True)

# Run Streamlit App
subprocess.run(["streamlit", "run", "disease_information_chatbot_app.py", "--server.port=8990"])

# Open ngrok tunnel for public URL
ngrok.set_auth_token(userdata.get('NGROK_API_KEY'))
ngrok_tunnel = ngrok.connect(8990)
print("Streamlit App:", ngrok_tunnel.public_url)
