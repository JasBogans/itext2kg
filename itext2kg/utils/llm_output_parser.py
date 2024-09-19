from typing import Union, List, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter
)
from langchain.schema import Document
from langchain.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.exceptions import OutputParserException
import time
import openai
import numpy as np
import tiktoken

class LangchainOutputParser:
    """
    A parser class for extracting and embedding information using LangChain's Text Splitters and OpenAI APIs.
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
        Initialize the LangchainOutputParser with specified models and operational parameters.

        Args:
            llm_model (ChatOpenAI): The language model instance.
            embeddings_model (Optional[OpenAIEmbeddings]): The embeddings model instance.
            text_splitter (Optional[TextSplitter]): The text splitter instance from LangChain. Defaults to MarkdownHeaderTextSplitter.
            sleep_time (int): Time to wait (in seconds) when encountering rate limits or errors.
            max_retries (int): Maximum number of retry attempts for handling errors.
            model_name (str): The model name for accurate token counting.
        """
        self.model = llm_model
        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.model_name = model_name

        # Initialize the text splitter; default to MarkdownHeaderTextSplitter if not provided
        if text_splitter is not None:
            self.text_splitter = text_splitter
        # else:
        #     self.text_splitter = MarkdownHeaderTextSplitter(
        #         headers_to_split_on=[
        #             ("#", "Header 1"),
        #             ("##", "Header 2"),
        #             ("###", "Header 3"),
        #         ],
        #         strip_headers=False  # Set to True to exclude headers from chunk content
        #     )
            
        # Initialize RecursiveCharacterTextSplitter for further splitting within chunks if necessary
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=30
        )

    def calculate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate embeddings for the given text using the initialized embeddings model.

        Args:
            text (Union[str, List[str]]): The text or list of texts to embed.

        Returns:
            np.ndarray: The calculated embeddings as a NumPy array.

        Raises:
            TypeError: If the input text is neither a string nor a list of strings.
        """
        if isinstance(text, list):
            return np.array(self.embeddings_model.embed_documents(text))
        elif isinstance(text, str):
            return np.array(self.embeddings_model.embed_query(text))
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text using tiktoken.

        Args:
            text (str): The text to estimate tokens for.

        Returns:
            int: Estimated number of tokens.
        """
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split the text into manageable chunks using the specified text splitter.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of text chunks.
        """
        documents = self.text_splitter.split_text(text)
        split_chunks = []
        for doc in documents:
            # Further split chunks that exceed the model's token limit
            if self.estimate_tokens(doc) > 3500:
                further_chunks = self.recursive_splitter.split_text(doc)
                split_chunks.extend(further_chunks)
            else:
                split_chunks.append(doc)
        return split_chunks

    def combine_results(self, results: List[dict]) -> dict:
        """
        Combine multiple JSON results into a single dictionary, merging values based on their types.

        Args:
            results (List[dict]): List of JSON objects.

        Returns:
            dict: Combined dictionary with merged values.
        """
        combined_dict = {}

        for d in results:
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

    def extract_information_as_json_for_context(
        self,
        output_data_structure,
        context: str,
        IE_query: str = '''
        # DIRECTIVES : 
        - Act like an experienced information extractor. 
        - If you do not find the right information, keep its place empty.
        '''
    ) -> Optional[dict]:
        """
        Extract information from a given context and format it as JSON using a specified structure.

        Args:
            output_data_structure: The data structure definition for formatting the JSON output.
            context (str): The context from which to extract information.
            IE_query (str): The query to provide to the language model for extracting information.

        Returns:
            Optional[dict]: The structured JSON output based on the provided data structure and extracted information.
                            Returns None if extraction fails.
        """
        retries = 0
        aggregated_results = []

        while retries < self.max_retries:
            try:
                # Step 1: Split text into chunks
                chunks = self.split_text_into_chunks(context)

                for chunk in chunks:
                    parser = JsonOutputParser(pydantic_object=output_data_structure)

                    template = f"""
                    Context: {chunk}

                    Question: {{query}}
                    Format_instructions : {{format_instructions}}
                    Answer: """

                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["query"],
                        partial_variables={"format_instructions": parser.get_format_instructions()},
                    )

                    # Chain the prompt, model, and parser
                    chain = prompt | self.model | parser

                    # Invoke the chain with the query
                    result = chain.invoke({"query": IE_query})

                    if result:
                        aggregated_results.append(result)

                # Combine all aggregated results into a single JSON object
                combined_result = self.combine_results(aggregated_results)
                return combined_result

            except openai.BadRequestError as e:
                if 'context_length_exceeded' in str(e).lower():
                    print(f"Context length exceeded: {e}. Attempting to truncate context.")
                    context = self.truncate_context(context)
                    retries += 1
                    time.sleep(self.sleep_time)
                else:
                    print(f"BadRequestError encountered: {e}.")
                    retries += 1
                    time.sleep(self.sleep_time)
            except openai.RateLimitError:
                print("Rate limit exceeded. Sleeping before retrying...")
                retries += 1
                time.sleep(self.sleep_time)
            except OutputParserException as e:
                print(f"OutputParserException encountered: {e}.")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}.")
                return None

        print("Max retries exceeded. Failed to extract information.")
        return None

    def truncate_context(self, context: str, max_tokens: int = 3500) -> str:
        """
        Truncate the context to ensure it does not exceed the maximum token limit.

        Args:
            context (str): The original context string.
            max_tokens (int): The maximum number of tokens allowed for the context.

        Returns:
            str: The truncated context string.
        """
        encoding = tiktoken.encoding_for_model(self.model_name)
        tokens = encoding.encode(context)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_context = encoding.decode(truncated_tokens)
            print(f"Truncated context to {max_tokens} tokens.")
            return truncated_context
        return context
