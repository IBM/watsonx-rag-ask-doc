from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
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
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

CONFIG = LoadConfig()


class Summarizer:

    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int,
    ):

        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        # max_summarizer_output_token = int(max_final_token / len(docs)) - token_threshold
        max_summarizer_output_token = max_final_token - token_threshold
        full_summary = ""
        counter = 1
        print("Generating the summary..")
        pdf_text = Summarizer.get_pdf_text(file_dir)
        text_chunk = Summarizer.get_text_chunk(pdf_text)
        summarizer_llm_system_role = summarizer_llm_system_role.format(
            max_summarizer_output_token
        )
        for chunk in text_chunk:
            prompt = chunk
            full_summary += Summarizer.get_llm_response(
                temperature, summarizer_llm_system_role, prompt=prompt
            )
        print("\nFull summary token length:", count_num_tokens(full_summary))
        final_summary = Summarizer.get_llm_response(
            temperature, final_summarizer_llm_system_role, prompt=full_summary
        )
        return final_summary

    @staticmethod
    def get_llm_response(temperature: float, llm_system_role: str, prompt: str):
        parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            max_new_tokens=CONFIG.max_token,
            min_new_tokens=30,
            temperature=temperature,
            top_k=CONFIG.k,
            top_p=0.9,
        )
        credentials = Credentials(
            api_key=CONFIG.genai_api_key,
            api_endpoint="https://bam-api.res.ibm.com/v2/text/chat?version=2024-03-19",
        )
        client = Client(credentials=credentials)
        model_id = CONFIG.genai_model_id
        response = client.text.chat.create(
            model_id=model_id,
            messages=[
                SystemMessage(
                    content=llm_system_role,
                ),
                HumanMessage(content=prompt),
            ],
            parameters=parameters,
        )
        return response.results[0].generated_text

    @staticmethod
    def get_text_chunk(text):
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    @staticmethod
    def get_pdf_text(file_path):
        pdf_reader = PdfReader(file_path)
        text = pdf_reader.pages[0].extract_text()
        return text
