"""Class for a VectorStore-backed memory object from langchain.
   Added path for return_messages parameter. Adds return_messages
   parameter and proper handling of that case to work with agent.
"""


from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.vectorstores.base import VectorStoreRetriever


class VectorStoreRetrieverMemory(BaseMemory):
    """Class for a VectorStore-backed memory object."""

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStoreRetriever object to connect to."""

    memory_key: str = "history"  #: :meta private:
    """Key name to locate the memories in the result of load_memory_variables."""

    input_key: Optional[str] = None
    """Key name to index the inputs to load_memory_variables."""

    return_docs: bool = False
    """Whether to return document objects."""

    return_messages: bool = False
    """Whether to return message objects."""

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys emitted from the load_memory_variables method."""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def load_memory_variables(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, Union[List[Document], str]]:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.get_relevant_documents(query)
        result: Union[List[Document], str]
        if self.return_docs:
            result = docs
        elif self.return_messages:
            result = []
            for doc in docs:
                # split lines in the page_content by '\n'
                msgs = doc.page_content.split('\n')
                for msg in msgs:
                    # assuming each msg is 'input' for user or 'output' for AI, instantiate
                    # HumanMessage or AIMessage accordingly, building list of messages
                    try:
                        role, content = msg.split(':')
                    except ValueError:
                        raise ValueError(f'Unexpected document format {msg}')
                    if role == 'input':
                        result.append(HumanMessage(content=content.lstrip()))
                    elif role == 'output':
                        result.append(AIMessage(content=content.lstrip()))
                    else:
                        raise ValueError(f'Unexpected message role {role}')
        else:
            result = "\n".join([doc.page_content for doc in docs])
        return {self.memory_key: result}

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        documents = self._form_documents(inputs, outputs)
        self.retriever.add_documents(documents)

    def clear(self) -> None:
        """Nothing to clear."""
