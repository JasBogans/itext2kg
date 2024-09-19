from typing import List, Optional
from ..utils import LangchainOutputParser  # Adjust the import path accordingly
from langchain.text_splitter import TextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class DocumentsDistiller:
    """
    A class designed to distill essential information from multiple documents into a combined
    structure, using natural language processing tools to extract and consolidate information.
    """
    def __init__(
        self,
        llm_model: ChatOpenAI,
        embeddings_model: Optional[OpenAIEmbeddings] = None,
        text_splitter: Optional[TextSplitter] = None,
        sleep_time: int = 5,
        max_retries: int = 3,
        model_name: str = "gpt-4"
    ) -> None:
        """
        Initializes the DocumentsDistiller with specified language model, embeddings model, and text splitter.

        Args:
            llm_model (ChatOpenAI): The language model instance to be used for generating semantic blocks.
            embeddings_model (Optional[OpenAIEmbeddings]): The embeddings model instance. Defaults to None.
            text_splitter (Optional[TextSplitter]): The text splitter instance from LangChain. If None, defaults to MarkdownHeaderTextSplitter.
            sleep_time (int): Time to wait (in seconds) when encountering rate limits or errors.
            max_retries (int): Maximum number of retry attempts for handling errors.
            model_name (str): The model name for accurate token counting.
        """
        self.langchain_output_parser = LangchainOutputParser(
            llm_model=llm_model,
            embeddings_model=embeddings_model,
            text_splitter=text_splitter,
            sleep_time=sleep_time,
            max_retries=max_retries,
            model_name=model_name
        )

    @staticmethod
    def __combine_dicts(dict_list: List[dict]) -> dict:
        """
        Combine a list of dictionaries into a single dictionary, merging values based on their types.

        Args:
            dict_list (List[dict]): A list of dictionaries to combine.

        Returns:
            dict: A combined dictionary with merged values.
        """
        combined_dict = {}

        for d in dict_list:
            for key, value in d.items():
                if key in combined_dict:
                    if isinstance(value, list) and isinstance(combined_dict[key], list):
                        combined_dict[key].extend(value)
                    elif isinstance(value, str) and isinstance(combined_dict[key], str):
                        if value and combined_dict[key]:
                            combined_dict[key] += f' {value}'
                        elif value:
                            combined_dict[key] = value
                    elif isinstance(value, dict) and isinstance(combined_dict[key], dict):
                        combined_dict[key].update(value)
                    else:
                        combined_dict[key] = value
                else:
                    combined_dict[key] = value

        return combined_dict

    def distill(
        self,
        documents: List[str],
        output_data_structure,
        IE_query: str
    ) -> dict:
        """
        Distill information from multiple documents based on a specific information extraction query.

        Args:
            documents (List[str]): A list of documents from which to extract information.
            output_data_structure: The data structure definition for formatting the output JSON.
            IE_query (str): The query to provide to the language model for extracting information.

        Returns:
            dict: A dictionary representing distilled information from all documents.
        """
        output_jsons = []
        for doc in documents:
            # Split the document into chunks using the provided TextSplitter
            chunks = self.langchain_output_parser.split_text_into_chunks(doc)

            for chunk in chunks:
                # Extract information from each chunk
                output_json = self.langchain_output_parser.extract_information_as_json_for_context(
                    context=chunk,  # Assuming 'split_text_into_chunks' returns strings
                    IE_query=IE_query,
                    output_data_structure=output_data_structure
                )
                if output_json:
                    output_jsons.append(output_json)

        # Combine all extracted JSON objects into a single dictionary
        return DocumentsDistiller.__combine_dicts(output_jsons)
