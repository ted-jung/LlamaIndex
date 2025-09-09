# =============================================================================
# chunking methods
# Created: 9, Sep 2025
# Updated: 9, Sep 2025
# Writer: Ted, Jung
# Description: Chunking methods for text data
# =============================================================================



# Sentence Splitter (LlamaIndex built-in) ===========


from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# Example text to be chunked
text = (
    "In the beginning, there was an idea. The idea was to create a new kind of "
    "language model. This model would be able to understand and generate text in a more "
    "human-like way.\n\n"
    "The first step was to gather a massive amount of data. This data was collected from "
    "books, articles, and websites all over the internet. Once the data was collected, "
    "it was preprocessed and cleaned to remove any noise or errors.\n\n"
    "The next step was to design the architecture of the model. The team decided to use "
    "a transformer-based architecture, which had shown great promise in previous NLP tasks."
)

# Initialize the Document object. LlamaIndex works with Document objects.
document = Document(text=text)

# Initialize the SentenceSplitter. It's similar to a recursive splitter.
# The 'chunk_size' and 'chunk_overlap' parameters control the splitting behavior.
text_splitter = SentenceSplitter(
    chunk_size=150,
    chunk_overlap=20,
)

# Split the document into nodes (chunks)
nodes = text_splitter.get_nodes_from_documents([document])

# Print the resulting chunks from the nodes
for i, node in enumerate(nodes):
    print(f"--- Chunk {i+1} (Length: {len(node.text)}) ---")
    print(node.text)



# LangChain Recursive Character Text Splitter ===========
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Example text to be chunked
text = (
    "In the beginning, there was an idea. The idea was to create a new kind of "
    "language model. This model would be able to understand and generate text in a more "
    "human-like way.\n\n"
    "The first step was to gather a massive amount of data. This data was collected from "
    "books, articles, and websites all over the internet. Once the data was collected, "
    "it was preprocessed and cleaned to remove any noise or errors.\n\n"
    "The next step was to design the architecture of the model. The team decided to use "
    "a transformer-based architecture, which had shown great promise in previous NLP tasks."
)

# Initialize the RecursiveCharacterTextSplitter
# The 'separators' list can be customized. The default is ["\n\n", "\n", " ", ""].
# 'chunk_size' is the maximum size of a chunk.
# 'chunk_overlap' creates overlap between chunks to maintain context.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=150,
    chunk_overlap=20,
    length_function=len
)

# Split the text into chunks
chunks = text_splitter.split_text(text)

# Print the resulting chunks
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
    print(chunk)



# Fixed-size chunking with overlap ===========
from typing import List
import re

# Split the text into units (words, in this case)
def word_splitter(source_text: str) -> List[str]:
    source_text = re.sub("\s+", " ", source_text)  # Replace multiple whitespces
    return re.split("\s", source_text)  # Split by single whitespace

def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float = 0.2) -> List[str]:
    text_words = word_splitter(text)
    overlap_int = int(chunk_size * overlap_fraction)
    chunks = []
    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    return chunks


# Recursive chunking
from typing import List

def recursive_chunking(text: str, max_chunk_size: int = 1000) -> List[str]
    # Base case: if text is small enough, return as single chunk
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Try separators in priority order
    separators = ["\n\n", "\n", ". ", " "]
    
    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # Check if adding this part would exceed the limit
                test_chunk = current_chunk + separator + part if current_chunk else part
                
                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            # Add the final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Recursively process any chunks that are still too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)
            
            return [chunk for chunk in final_chunks if chunk]
    
    # Fallback: split by character limit if no separators work
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


# Document based chunking

from typing import List
import re

def markdown_document_chunking(text: str) -> List[str]:
    # Split by markdown headers (# ## ### etc.)
    header_pattern = r'^#{1,6}\s+.+$'
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    
    for line in lines:
        # Check if this line is a header
        if re.match(header_pattern, line, re.MULTILINE):
            # Save previous chunk if it has content
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            
            # Start new chunk with this header
            current_chunk = [line]
        else:
            # Add line to current chunk
            current_chunk.append(line)
    
    # Add final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return chunks



