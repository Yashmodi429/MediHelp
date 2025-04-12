#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain==0.1.12 -q')
get_ipython().system('pip install langchain-google-genai==0.0.7 -q')
get_ipython().system('pip install langchain-community==0.0.29 -q')
get_ipython().system('pip install streamlit==1.32.2 -q')
get_ipython().system('pip install pyngrok==7.1.5 -q')
get_ipython().system('pip install google-generativeai>=0.3.2 -q')


# In[73]:


get_ipython().system('pip install streamlit streamlit_webrtc googletrans==4.0.0-rc1 langchain langchain-google-genai')
get_ipython().system('pip install SpeechRecognition pyaudio')



# In[77]:


import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GEMINI_API_KEY')


# In[132]:


get_ipython().run_cell_magic('writefile', 'disease_information_chatbot_app.py', '\nfrom langchain_google_genai import ChatGoogleGenerativeAI\nfrom langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder\nfrom langchain_community.chat_message_histories import StreamlitChatMessageHistory\nfrom langchain_core.runnables.history import RunnableWithMessageHistory\nimport streamlit as st\n\n# ---------- Page Config ----------\nst.set_page_config(page_title="AI Medical Assistant 🤖", page_icon="🤖", layout="wide")\n\n# ---------- Custom CSS for styling ----------\nst.markdown("""\n    <style>\n        /* Global styles */\n        body {\n            background-color: white; /* White background */\n            font-family: \'Arial\', sans-serif;\n            color: #333;\n        }\n\n        .header-title {\n            font-size: 40px;\n            text-align: center;\n            color: #D32F2F; /* Red color */\n            margin-top: 30px;\n            font-weight: bold;\n        }\n\n        .header-subtitle {\n            font-size: 18px;\n            text-align: center;\n            color: #555; /* Grey color */\n            margin-top: 10px;\n        }\n\n        /* Chatbox styling */\n        .chat-box {\n            background-color: #fff;\n            border-radius: 15px;\n            padding: 20px;\n            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);\n            margin-bottom: 50px;\n            height: 400px;\n            overflow-y: scroll;\n        }\n\n        /* Footer styling */\n        .footer {\n            text-align: center;\n            padding: 15px;\n            background-color: #D32F2F; /* Red background */\n            color: white;\n            font-size: 16px;\n            position: fixed;\n            width: 100%;\n            bottom: 0;\n            font-weight: bold;\n        }\n\n        /* Chat message styling */\n        .chat-message {\n            margin-bottom: 15px;\n        }\n        .user-message {\n            background-color: #DCF8C6;\n            padding: 12px;\n            border-radius: 10px;\n            width: fit-content;\n            max-width: 75%;\n            font-size: 16px;\n            color: #333;\n        }\n        .ai-message {\n            background-color: #EAEAEA;\n            padding: 12px;\n            border-radius: 10px;\n            width: fit-content;\n            max-width: 75%;\n            margin-left: auto;\n            font-size: 16px;\n            color: #333;\n        }\n\n        /* Text input field styling */\n        .stTextInput input {\n            font-size: 16px;\n        }\n\n        .stChatInput {\n            font-size: 16px;\n        }\n\n        /* Bootstrap styling for container and chat input */\n        .chat-container {\n            display: flex;\n            flex-direction: column;\n            justify-content: space-between;\n            height: 80vh;\n        }\n\n        .chat-content {\n            flex-grow: 1;\n            overflow-y: auto;\n        }\n\n        .chat-input-container {\n            display: flex;\n            margin-top: 20px;\n        }\n\n        .chat-input {\n            width: 100%;\n            padding: 10px;\n            font-size: 16px;\n            border-radius: 5px;\n            border: 1px solid #ccc;\n            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);\n        }\n\n        .highlighted-term {\n            background-color: #4CAF50;\n            color: white;\n            padding: 2px 8px;\n            border-radius: 5px;\n            cursor: pointer;\n        }\n    </style>\n""", unsafe_allow_html=True)\n\n# ---------- Header ----------\nst.markdown(\'<div class="header-title">🤖 Welcome to Your AI Medical Assistant</div>\', unsafe_allow_html=True)\nst.markdown(\'<div class="header-subtitle">Developed with care by Yash Samir Modi</div>\', unsafe_allow_html=True)\n\n# ---------- Gemini Setup ----------\ngemini = ChatGoogleGenerativeAI(\n    model=\'gemini-2.0-flash-001\',\n    temperature=0.1,\n    convert_system_message_to_human=True\n)\n\n# Custom prompt to generate the desired output format for any medical condition\nSYS_PROMPT = """\nYou are a medical assistant designed to provide clear, accurate, and up-to-date information about various medical conditions.\n\nWhen given a medical condition, respond with the following format:\n\nCondition Name: {{Condition Name}}\nOverview: {{Brief summary of the condition}}\n\nCommon Symptoms:\n{{List of common symptoms}}\n\nCauses:\n{{List of common causes or risk factors}}\n\nDiagnosis:\n{{How it is usually diagnosed}}\n\nTreatment Options:\n{{Medications, therapies, or procedures}}\n\nPrevention Tips:\n{{How to prevent it, if possible}}\n\nWhen to See a Doctor:\n{{Red flags or emergency signs}}\n\nSources:\n{{Trusted source references like Mayo Clinic, WebMD, WHO}}\n\nIf the user asks about the specific aspects of the condition like "Diagnosis," "Symptoms," or "Treatment," provide relevant information for those subcategories.\nFor example, if the user asks about "Diabetes," respond with structured information on Diabetes. Similarly, if the user asks about "Hypertension," "Asthma," or any other medical condition, follow the same format.\n"""\n\nprompt = ChatPromptTemplate.from_messages(\n    [\n        ("system", SYS_PROMPT),\n        MessagesPlaceholder(variable_name="history"),\n        ("human", "{input}"),\n    ]\n)\n\nllm_chain = prompt | gemini\nstreamlit_msg_history = StreamlitChatMessageHistory()\n\nconversation_chain = RunnableWithMessageHistory(\n    llm_chain,\n    lambda session_id: streamlit_msg_history,\n    input_messages_key="input",\n    history_messages_key="history",\n)\n\n# ---------- Chat History Box ----------\nst.markdown(\'<div class="chat-container">\', unsafe_allow_html=True)\n\nst.markdown(\'<div class="chat-content">\', unsafe_allow_html=True)\nfor msg in streamlit_msg_history.messages:\n    with st.chat_message(msg.type):\n        if msg.type == "human":\n            st.markdown(f\'<div class="user-message">{msg.content}</div>\', unsafe_allow_html=True)\n        else:\n            # Highlight medical terms in AI\'s message\n            response = msg.content\n            terms_to_highlight = ["Diabetes", "Insulin", "Symptoms", "Treatment", "Diagnosis", "Hypertension", "Cancer", "Asthma"]  # Add more terms\n            for term in terms_to_highlight:\n                response = response.replace(term, f\'<span class="highlighted-term">{term}</span>\')\n            st.markdown(f\'<div class="ai-message">{response}</div>\', unsafe_allow_html=True)\nst.markdown(\'</div>\', unsafe_allow_html=True)\n\n# ---------- Chat Input ----------\nst.markdown(\'<div class="chat-input-container">\', unsafe_allow_html=True)\nuser_prompt = st.chat_input("Type your medical condition here...")\nif user_prompt:\n    st.chat_message("human").markdown(user_prompt)\n    with st.chat_message("ai"):\n        try:\n            config = {"configurable": {"session_id": "any"}}\n            response = conversation_chain.invoke({"input": user_prompt}, config)\n            if hasattr(response, "content"):\n                # Highlight terms in the AI response for possible further questioning\n                response_content = response.content\n                terms_to_highlight = ["Diabetes", "Insulin", "Symptoms", "Treatment", "Diagnosis"]\n                for term in terms_to_highlight:\n                    response_content = response_content.replace(term, f\'<span class="highlighted-term">{term}</span>\')\n                st.markdown(f\'<div class="ai-message">{response_content}</div>\', unsafe_allow_html=True)\n            else:\n                st.markdown(f\'<div class="ai-message">{str(response)}</div>\', unsafe_allow_html=True)\n        except Exception as e:\n            st.error(f"⚠️ Something went wrong: {e}")\nst.markdown(\'</div>\', unsafe_allow_html=True)\n\n# ---------- Footer ----------\nst.markdown(\'<div class="footer">Developed by Yash Samir Modi | Medical Assistant Chatbot | Powered by AI</div>\', unsafe_allow_html=True)\n')


# In[133]:


get_ipython().system('streamlit run disease_information_chatbot_app.py --server.port=8990 &>./logs.txt &')


# In[134]:


from pyngrok import ngrok

# Terminate open tunnels if exist
ngrok.kill()

ngrok.set_auth_token(userdata.get('NGROK_API_KEY'))

# Open an HTTPs tunnel on port XXXX which you get from your `logs.txt` file
ngrok_tunnel = ngrok.connect(8990)
print("Streamlit App:", ngrok_tunnel.public_url)


# In[129]:


ngrok.kill()


# In[130]:


get_ipython().system('ps -ef | grep streamlit')


# In[131]:


get_ipython().system('sudo kill -9 14111')

