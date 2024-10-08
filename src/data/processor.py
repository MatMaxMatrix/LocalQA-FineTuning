"""Document processing utilities for different file formats."""
#%%
from abc import ABC, abstractmethod
from typing import List, Any
import logging
from pathlib import Path
import re
from dataclasses import dataclass
import yaml
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
#import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
print(logger)
#%%

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    
    @classmethod
    def from_config(cls, config_path: str) -> 'ProcessingConfig':
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        data_config = conf.get('data', {})
        return cls(
            chunk_size=data_config.get('chunk_size', 2000),
            chunk_overlap=data_config.get('chunk_overlap', 100),
            separators=data_config.get('separators', ["\n\n", "\n", ". ", "! ", "? "])
        )
        

#%%
class ProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, file_path: str, config: ProcessingConfig) -> None:
        """Initialize document processor.
        
        Args:
            file_path: Path to the document file
            config: Processing configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=config.separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    @abstractmethod
    def process(self) -> Dataset:
        """Process the document and return a dataset.
        
        Returns:
            Dataset containing processed text chunks
            
        Raises:
            ProcessingError: If processing fails
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove reference numbers
        text = re.sub(r'\[\d+\]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Ensure proper sentence endings
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
            
        return text

class WordProcessor(DocumentProcessor):
    """Processor for Word documents."""
    
    def process(self) -> Dataset:
        """Process Word document.
        
        Returns:
            Dataset containing processed text chunks
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            doc = Document(self.file_path)
            combined_text = []
            
            for paragraph in doc.paragraphs:
                if text := paragraph.text.strip():
                    combined_text.append(self._clean_text(text))
                    
            text_blocks = self.text_splitter.create_documents(['\n'.join(combined_text)])
            
            return Dataset.from_list([
                {"text": block.page_content} 
                for block in text_blocks
            ])
            
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            raise ProcessingError(f"Failed to process Word document: {e}")

class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents."""
    
    def process(self) -> Dataset:
        """Process PDF document.
        
        Returns:
            Dataset containing processed text chunks
            
        Raises:
            ProcessingError: If processing fails
        """
        try:
            doc = fitz.open(self.file_path)
            combined_text = []
            
            for page in doc:
                if text := page.get_text().strip():
                    combined_text.append(self._clean_text(text))
                    
            text_blocks = self.text_splitter.create_documents(['\n'.join(combined_text)])
            
            return Dataset.from_list([
                {"text": block.page_content} 
                for block in text_blocks
            ])
            
        except Exception as e:
            logger.error(f"Error processing PDF document: {e}")
            raise ProcessingError(f"Failed to process PDF document: {e}")

def get_processor(
    file_path: str, 
    file_type: str, 
    config: ProcessingConfig = None
) -> DocumentProcessor:
    """Factory function to get appropriate document processor.
    
    Args:
        file_path: Path to document file
        file_type: Type of document ('word' or 'pdf')
        config: Optional processing configuration
        
    Returns:
        DocumentProcessor instance
        
    Raises:
        ValueError: If file type is not supported
    """
    if config is None:
        config = ProcessingConfig()
        
    processors = {
        'word': WordProcessor,
        'pdf': PDFProcessor
    }
    
    if file_type not in processors:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    return processors[file_type](file_path, config)
