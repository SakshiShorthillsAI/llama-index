from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.text_splitter import SentenceSplitter
from llama_index.llms import OpenAI, AzureOpenAI
from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.schema import IndexNode
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding, resolve_embed_model
from llama_index import download_loader, set_global_service_context, StorageContext, load_index_from_storage
from llama_index.callbacks import CallbackManager, TokenCountingHandler
import chromadb
import pandas as pd
import tiktoken
import os
import re
from typing import List


os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_API_VERSION"] = ""


class LlamaProcessor:
    def __init__(self, llm_config, file_path):
        self.llm_config = llm_config
        self.file_path = file_path
        self.slides_parser = SimpleNodeParser.from_defaults(
            include_prev_next_rel=True,
            include_metadata=True
        )
        self.bullet_node_parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=self.custom_sentence_splitter,
            window_size=3,
            include_prev_next_rel=True,
            include_metadata=True
        )

    def custom_sentence_splitter(self, text: str) -> List[str]:
        return re.split(r'\n●|\n-|\n', text)

    def load_documents(self):
        """Load documents from the specified Excel file."""
        pandas_excel_reader = download_loader("PandasExcelReader")
        loader = pandas_excel_reader(pandas_config={"header": 0})
        return loader.load_data(file=Path(self.file_path))

    def process_nodes(self, nodes, sub_node_parsers):
        """Process nodes using sub-node parsers."""
        all_nodes = []
        for base_node in nodes:
            for parser in sub_node_parsers:
                sub_nodes = parser.get_nodes_from_documents([base_node])
                sub_inodes = [IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes]
                all_nodes.extend(sub_inodes)

            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        return all_nodes

    def run_llama(self):
        """Run the Llama process."""
        llm = AzureOpenAI( model_kwargs={
                "headers":
                    {
                    "User-Id": "SHT-PWC-Project-8735"
                    }
                                },
    engine="GPT35", model="gpt-35-turbo", temperature=0.0)

        set_global_service_context(ServiceContext.from_defaults(
            llm=llm,
            callback_manager=CallbackManager([TokenCountingHandler(
                tokenizer=tiktoken.encoding_for_model("gpt-35-turbo").encode)]),
            embed_model=resolve_embed_model("local:BAAI/bge-small-en-v1.5")
        ))

        vector_index_chunk = VectorStoreIndex(self.all_nodes, service_context=ServiceContext.from_defaults())
        vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=8)
        retriever_chunk = RecursiveRetriever("vector", retriever_dict={"vector": vector_retriever_chunk},
                                            node_dict=self.all_nodes_dict, verbose=True)
        query_engine_chunk = RetrieverQueryEngine.from_args(retriever_chunk,
                                                            service_context=ServiceContext.from_defaults(),
                                                            verbose=True, response_mode="compact")
        return query_engine_chunk

    def process_excel_data(self, query_engine_chunk):
        """Process data from the Excel file using Llama."""
        df = pd.read_excel(self.file_path)
        data_points = df['Data Points'].to_list()
        financial_explanation = df['Finance explanation / User story'].to_list()
        ques = df['SHAI Questions Simplified'].to_list()
        final_value = []

        for data_points_val, financial_explanation_val, ques_val in zip(data_points, financial_explanation, ques):
            result = []

            if not pd.isna(ques_val):
                questions_with_numbers = re.findall(r'\d+\.\s*([^?]+)', ques_val)

                for question_with_number in questions_with_numbers:
                    simplified_ques = question_with_number.strip()
                    response = query_engine_chunk.query(simplified_ques)
                    ans = response.response
                    result.append(ans)
                    result.append("////")

            final_data = {'Data Points': data_points_val,
                          'Financial Explanation/User Story': financial_explanation_val,
                          'Sht simplified questiosn': ques_val,
                          'Results': result
                          }
            final_value.append(final_data)

        return pd.DataFrame(final_value)

    def run(self):
        """Run the complete Llama processing."""
        docs0 = self.load_documents()
        self.slides_nodes = self.slides_parser.get_nodes_from_documents(docs0)
        self.all_nodes = self.process_nodes(nodes=self.slides_nodes, sub_node_parsers=[self.bullet_node_parser])
        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}
        query_engine_chunk = self.run_llama()
        result_dataframe = self.process_excel_data(query_engine_chunk)
        return result_dataframe


if __name__ == "__main__":
    # Configure Llama
    llm_config = {
        "model_kwargs": {"headers": {"User-Id": "SHT-PWC-Project-8735"}},
        "embed_model": resolve_embed_model("local:BAAI/bge-small-en-v1.5"),
        "callback_manager": CallbackManager([TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("gpt-35-turbo").encode)])
    }

    # Provide file path
    file_path = 'data/Book 1_2.xlsx'

    # Initialize and run LlamaProcessor
    llama_processor = LlamaProcessor(llm_config=llm_config, file_path=file_path)
    result_df = llama_processor.run()

    # Save results to Excel
    result_df.to_excel('dubai_1_result.xlsx', index=False)
