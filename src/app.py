import streamlit as st
import os
from retriever import Retriever
from generator import Generator
from utils import get_readme
import requests

default_model = "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net"
default_llm = "llama3"  # Default Ollama model

st.set_page_config(layout="wide")
st.title("Chat with your code")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Ollama configuration
    st.subheader("Ollama Settings")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434", 
                            help="The URL where your Ollama server is running")
    
    # Model selection from common Ollama models
    llm_model = st.selectbox(
        "Select Ollama Model",
        options=["llama3", "codellama", "llama2", "mistral", "mixtral", "phi"],
        index=0,
        help="Choose which Ollama model to use for code understanding and generation"
    )
    
    mode = st.radio(
        "Interaction Mode",
        ["Understand", "Generate"],
        help="Understand: Explain existing code | Generate: Create new code"
    )
    
    # Add a button to check Ollama status
    if st.button("Check Ollama Status"):
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = [model.get("name") for model in response.json().get("models", [])]
                if llm_model in available_models:
                    st.success(f"✅ Ollama server is running and {llm_model} is available")
                else:
                    st.warning(f"⚠️ Ollama server is running but {llm_model} is not found. Available models: {', '.join(available_models)}")
                    st.info(f"To download {llm_model}, run: `ollama pull {llm_model}`")
            else:
                st.error("❌ Connected to Ollama server but received an error response")
        except Exception as e:
            st.error(f"❌ Couldn't connect to Ollama server at {ollama_url}. Error: {str(e)}")
            st.info("Make sure Ollama is installed and running. Visit https://ollama.com for installation instructions.")

user_repo = st.text_input(
    "Github Link to a public codebase.", 
    placeholder="Enter the HTTPS link to your github repo here",
)

if user_repo:
    # Load the GitHub Repo
    st.write("Input Repo: ", user_repo)
    retriever = Retriever(user_repo, default_model)
    st.write("Your repo has been successfully cloned")

    # Initialize generator
    generator = Generator(
        model_name=llm_model,
        ollama_base_url=ollama_url
    )

    # Chunk and Create Vector DB
    st.write("Parsing Repository content and creating Vector Embeddings. This may take a while..")
    retriever.load_db()
    st.write("Done Loading. Ready to help with your queries!")

    repo_readme = get_readme(retriever.clone_path)
    if repo_readme:
        with st.container(border=True):
            st.write("**Here is the README of the repository to help you get started!**")
            st.markdown(repo_readme)

    # Maintain chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "retrieval":
                for i, result in enumerate(message["content"]):
                    st.text(f"Result {i+1}: {result['file_name']}")
                    st.code(result["page_content"])
            else:  # assistant
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your question here."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant code snippets
        response = retriever.retrieve_results(prompt)
        
        # Display retrieved results
        with st.chat_message("retrieval"):
            st.write("Here are the top 3 results")
            cols = st.columns(len(response), vertical_alignment="top")
            for i in range(len(response)):
                with cols[i]:
                    st.text(f"Result {i+1}: {response[i]['file_name']}")
                    st.code(response[i]["page_content"])
        
        # Add retrieval results to chat history
        st.session_state.messages.append({"role": "retrieval", "content": response})
        
        # Show a spinner while generating the response
        with st.spinner("Generating response with Ollama..."):
            # Check Ollama status before attempting to generate
            status = generator.check_ollama_status()
            
            if status["status"] == "available":
                process_mode = "generate" if mode == "Generate" else "understand"
                generated_response = generator.process_query(prompt, response, mode=process_mode.lower())
                
                # Display generated response
                with st.chat_message("assistant"):
                    st.markdown(generated_response)
                
                # Add generated response to chat history
                st.session_state.messages.append({"role": "assistant", "content": generated_response})
            else:
                # Display error message if Ollama is not available
                with st.chat_message("assistant"):
                    st.error(f"Ollama error: {status['message']}")
                    st.info("Please make sure Ollama is installed and running. Visit https://ollama.com for installation instructions.")
                    if status["status"] == "model_not_found":
                        st.info(f"To download the {llm_model} model, run: `ollama pull {llm_model}`")
