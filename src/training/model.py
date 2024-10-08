"""Model management utilities for fine-tuning and inference."""

import logging
from typing import Optional, List, Tuple
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelManager:
    """Manager for handling model loading, preparation, and saving."""
    
    def __init__(
        self,
        config_path: str
    ):
        """Initialize model manager."""
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        model_config = conf.get('model', {})
        
        self.model_name = model_config.get('base_model', 'unsloth/llama-3-8b-bnb-4bit')
        self.max_seq_length = model_config.get('max_seq_length', 2048)
        self.load_in_4bit = model_config.get('load_in_4bit', True)
        self.dtype = model_config.get('dtype')
        if self.dtype is None:
            self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None


    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ModelError: If loading fails
        """
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")

    def prepare_for_training(
        self,
        config_path: str
    ):
        """Prepare model for training with LoRA."""
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        model_config = conf.get('model', {})
        
        target_modules = model_config.get('target_modules', ["o_proj"])
        lora_r = model_config.get('lora_r', 16)
        lora_alpha = model_config.get('lora_alpha', 16)
        use_gradient_checkpointing = model_config.get('use_gradient_checkpointing', True)

        if self.model is None:
            raise ModelError("Model must be loaded before preparation")
            
        try:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora_r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth" if use_gradient_checkpointing else False,
                random_state=3407
            )
            
            logger.info("Successfully prepared model for training")
            return self.model
            
        except Exception as e:
            logger.error(f"Error preparing model: {e}")
            raise ModelError(f"Failed to prepare model: {e}")

    def save_model(
        self,
        output_dir: str,
        quantization_method: str = "q8_0"
    ) -> None:
        """Save the model in GGUF format.
        
        Args:
            output_dir: Directory to save the model
            quantization_method: Quantization method to use
            
        Raises:
            ModelError: If saving fails
        """
        if self.model is None or self.tokenizer is None:
            raise ModelError("Model and tokenizer must be loaded before saving")
            
        try:
            self.model.save_pretrained_gguf(
                output_dir,
                self.tokenizer,
                quantization_method=quantization_method
            )
            
            logger.info(f"Successfully saved model to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ModelError(f"Failed to save model: {e}")