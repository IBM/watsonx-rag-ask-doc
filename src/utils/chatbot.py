import gradio as gr
import time
import os
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
from utils.load_config import LoadConfig

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import Optional
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    ModerationHAP,
    ModerationParameters,
    SystemMessage,
    TextGenerationParameters,
)

APPCFG = LoadConfig()
URL = "https://github.ibm.com/kirtijha/WatsonX-RAG-Ask-Doc.git"
hyperlink = f"[ASK-DOC user guideline]({URL})"


credentials = Credentials(api_key=APPCFG.genai_api_key)
client = Client(credentials=credentials)
embeddings = LangChainEmbeddingsInterface(
    client=client,
    model_id="sentence-transformers/all-minilm-l6-v2",
    parameters=TextEmbeddingParameters(truncate_input_tokens=True),
)


class ChatBot:
    @staticmethod
    def respond(
        chatbot: List,
        message: str,
        data_type: str = "Preprocessed doc",
        temperature: float = 0.0,
    ) -> Tuple:
        if data_type == "Preprocessed doc":
            # directories
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(
                    persist_directory=APPCFG.persist_directory,
                    embedding_function=embeddings,
                )
            else:
                chatbot.append(
                    (
                        message,
                        f"VectorDB does not exist. Please first execute the 'Preprocess files'. For further information please visit {hyperlink}.",
                    )
                )
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(
                    persist_directory=APPCFG.custom_persist_directory,
                    embedding_function=embeddings,
                )
            else:
                chatbot.append(
                    (
                        message,
                        f"No file was uploaded. Please first upload your files using the 'upload' button.",
                    )
                )
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        print(docs)
        question = "# User new question:\n" + message
        retrieved_content = ChatBot.clean_references(docs)
        # Memory: previous two Q&A pairs
        chat_history = (
            f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        )
        prompt = f"{chat_history}{retrieved_content}{question}"
        print("========================")
        print(prompt)
        parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            max_new_tokens=APPCFG.max_token,
            min_new_tokens=30,
            temperature=APPCFG.temperature,
            top_k=APPCFG.k,
            top_p=0.9,
        )
        credentials = Credentials(
            api_key=APPCFG.genai_api_key,
            api_endpoint="https://bam-api.res.ibm.com/v2/text/chat?version=2024-03-19",
        )
        client = Client(credentials=credentials)
        model_id = APPCFG.genai_model_id
        response = client.text.chat.create(
            model_id=model_id,
            messages=[
                SystemMessage(
                    content=APPCFG.llm_system_role,
                ),
                HumanMessage(content=prompt),
            ],
            parameters=parameters,
        )
        chatbot.append((message, response.results[0].generated_text))
        time.sleep(2)

        return "", chatbot, retrieved_content

    @staticmethod
    def clean_references(documents: List) -> str:
        server_url = "http://localhost:8000"
        documents = [str(x) + "\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            # Extract content and metadata
            content, metadata = re.match(
                r"page_content=(.*?)( metadata=\{.*\})", doc
            ).groups()
            metadata = metadata.split("=", 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            # Decode newlines and other escape sequences
            content = bytes(content, "utf-8").decode("unicode_escape")

            # Replace escaped newlines with actual newlines
            content = re.sub(r"\\n", "\n", content)
            # Remove special tokens
            content = re.sub(r"\s*<EOS>\s*<pad>\s*", " ", content)
            # Remove any remaining multiple spaces
            content = re.sub(r"\s+", " ", content).strip()

            # Decode HTML entities
            content = html.unescape(content)

            # Replace incorrect unicode characters with correct ones
            content = content.encode("latin1").decode("utf-8", "ignore")

            # Remove or replace special characters and mathematical symbols
            # This step may need to be customized based on the specific symbols in your documents
            content = re.sub(r"â", "-", content)
            content = re.sub(r"â", "∈", content)
            content = re.sub(r"Ã", "×", content)
            content = re.sub(r"ï¬", "fi", content)
            content = re.sub(r"â", "∈", content)
            content = re.sub(r"Â·", "·", content)
            content = re.sub(r"ï¬", "fl", content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += (
                f"# Retrieved content {counter}:\n"
                + content
                + "\n\n"
                + f"Source: {os.path.basename(metadata_dict['source'])}"
                + " | "
                + f"Page number: {str(metadata_dict['page'])}"
                + "\n\n"
            )
            counter += 1

        return markdown_documents
