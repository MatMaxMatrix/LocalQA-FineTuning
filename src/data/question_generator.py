"""Question generation module for creating training data."""
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import requests
from datasets import Dataset
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for question generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 500
    stop_sequences: List[str] = None

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["<|im_end|>", "</s>"]

class QuestionGenerationError(Exception):
    """Base exception for question generation errors."""
    pass

class QuestionGenerator:
    """Generator for creating question-answer pairs from documents."""
    def __init__(
        self, 
        config_path: str
    ):
        """Initialize question generator."""
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        ollama_config = conf.get('ollama', {})
        self.uri = f'http://{ollama_config.get("host", "localhost:11434")}/api/generate'
        self.model = ollama_config.get('model', 'llama3.2:latest')
        self.config = GenerationConfig.from_config(config_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _generate_questions(self, paragraph: str) -> List[Dict[str, str]]:
        """Generate questions for a given paragraph.
        
        Args:
            paragraph: Text to generate questions from
            
        Returns:
            List of question-answer pairs
            
        Raises:
            QuestionGenerationError: If generation fails
        """
        prompt = f"""
        Given the following paragraph, generate relevant technical questions
        that a user might ask about the information presented. For each question,
        provide a concise answer excerpted directly from the text.

        Present the results in JSON format with "question" and "answer" fields.
        Only respond with the JSON array, no additional text.

        Paragraph:
        {paragraph}
        """
        
        try:
            response = requests.post(
                self.uri,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "stop": self.config.stop_sequences
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return json.loads(result['response'])
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise QuestionGenerationError(f"Failed to generate questions: {e}")

    def prepare_training_data(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Prepare training data from dataset.
        
        Args:
            dataset: Dataset containing text chunks
            
        Returns:
            List of training examples
            
        Raises:
            QuestionGenerationError: If preparation fails
        """
        training_data = []
        
        for item in dataset:
            try:
                qa_pairs = self._generate_questions(item['text'])
                
                for qa in qa_pairs:
                    example = {
                        "instruction": qa['question'].strip(),
                        "input": "",
                        "output": f"Based on the document: {qa['answer'].strip()}"
                    }
                    training_data.append(example)
                    
            except Exception as e:
                logger.warning(f"Skipping problematic text chunk: {e}")
                continue
                
        if not training_data:
            raise QuestionGenerationError("No training examples were generated")
            
        return training_data