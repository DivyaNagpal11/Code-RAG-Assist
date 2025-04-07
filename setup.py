from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="code-rag",
    version="0.1.0",
    description='A simple RAG application to chat wth your code',
    packages=find_packages(),
    author='Divya Nagpal',
    author_email='your.email@example.com',
    install_requires=[
        "streamlit>=1.24.0",
        "llama-index>=0.9.13",
        "chromadb>=0.4.18",
        "tqdm>=4.66.1",
        "gitpython>=3.1.41",
        "torch>=2.0.0",
        "langchain>=0.1.4",
        "langchain-core>=0.1.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.10",
)