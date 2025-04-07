import os
from typing import List, Dict, Optional
import requests
import json

from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Generator:
    """
    The Generator class handles all logic for LLM-based response generation
    using retrieved code snippets with Ollama local models.
    """

    def __init__(
        self,
        model_name: str = "llama3:latest",  # Default to llama3 but can use codellama, mixtral, etc.
        temperature: float = 0.2,
        max_tokens: int = 2048,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Generator with the specified Ollama LLM settings.
        
        :param model_name: Name of the Ollama model to use (llama3, codellama, etc.)
        :param temperature: Temperature for generation (lower means more deterministic)
        :param max_tokens: Maximum tokens for generation
        :param ollama_base_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_base_url = ollama_base_url
        
        # Initialize LLM with Ollama
        self.llm = Ollama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            base_url=self.ollama_base_url
        )
        
        # Create prompt template for code understanding
        self.code_understanding_prompt = ChatPromptTemplate.from_template(
            """You are an expert developer who helps understand codebases and answer questions about them.
            
            Here are relevant code snippets that might help answer the user's question:
            
            {context}
            
            User's question: {question}
            
            Based on the code snippets provided, answer the user's question thoroughly.
            Include specific code references when applicable.
            If the code snippets don't contain enough information to answer fully, acknowledge that limitation.
            """
        )
        
        # Create prompt template for code generation
        self.code_generation_prompt = ChatPromptTemplate.from_template(
            """You are an expert developer who helps generate code based on existing codebases.
            
            Here are relevant code snippets from the user's codebase:
            
            {context}
            
            User's request: {question}
            
            Generate appropriate code that aligns with the existing codebase style, structure, and conventions.
            Provide clear explanations about your implementation choices.
            """
        )
        
        # Set up chains for understanding and generation
        self.understanding_chain = (
            self.code_understanding_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        self.generation_chain = (
            self.code_generation_prompt 
            | self.llm 
            | StrOutputParser()
        )

    def check_ollama_status(self) -> Dict:
        """
        Check if Ollama server is running and the requested model is available.
        
        :return: Status information dictionary
        """
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                # Check if model name exists exactly or as a prefix
                model_available = any(name.startswith(self.model_name) for name in model_names)
                
                return {
                    "status": "available" if model_available else "model_not_found",
                    "available_models": model_names,
                    "message": f"Model '{self.model_name}' ready" if model_available else f"Model '{self.model_name}' not found. Available models: {model_names}"
                }
            else:
                return {"status": "server_error", "message": "Error connecting to Ollama server"}
        except requests.exceptions.ConnectionError:
            return {"status": "unavailable", "message": "Ollama server is not running"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _format_context(self, retrieved_results: List[Dict]) -> str:
        """
        Format retrieved code snippets into a context string for the LLM.
        
        :param retrieved_results: List of dictionaries with code snippets and filenames
        :return: Formatted context string
        """
        context = ""
        for i, result in enumerate(retrieved_results, 1):
            context += f"SNIPPET {i} (from {result['file_name']}):\n```\n{result['page_content']}\n```\n\n"
        return context

    def understand_code(self, query: str, retrieved_results: List[Dict]) -> str:
        """
        Generate a response that helps understand the retrieved code snippets
        based on the user's query.
        
        :param query: User's question or query
        :param retrieved_results: List of retrieved code snippets
        :return: Generated explanation/response
        """
        # Check Ollama status before proceeding
        status = self.check_ollama_status()
        if status["status"] != "available":
            return f"Error: {status['message']}"
            
        context = self._format_context(retrieved_results)
        try:
            response = self.understanding_chain.invoke({
                "context": context,
                "question": query
            })
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_code(self, query: str, retrieved_results: List[Dict]) -> str:
        """
        Generate new code based on the user's request and retrieved code snippets.
        
        :param query: User's code generation request
        :param retrieved_results: List of retrieved code snippets for context
        :return: Generated code with explanation
        """
        # Check Ollama status before proceeding
        status = self.check_ollama_status()
        if status["status"] != "available":
            return f"Error: {status['message']}"
            
        context = self._format_context(retrieved_results)
        try:
            response = self.generation_chain.invoke({
                "context": context,
                "question": query
            })
            return response
        except Exception as e:
            return f"Error generating code: {str(e)}"
    
    def process_query(self, query: str, retrieved_results: List[Dict], mode: str = "understand") -> str:
        """
        Process a user query by either understanding existing code or generating new code.
        
        :param query: User's question or request
        :param retrieved_results: List of retrieved code snippets
        :param mode: Processing mode - "understand" or "generate"
        :return: Generated response
        """
        if mode == "generate":
            return self.generate_code(query, retrieved_results)
        else:  # Default to understand mode
            return self.understand_code(query, retrieved_results)
        
