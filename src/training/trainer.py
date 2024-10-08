"""Training utilities for fine-tuning models."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    """Configuration for training."""
    output_dir: str
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    max_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: Optional[int]
    warmup_steps: int
    weight_decay: float
    seed: int

    @classmethod
    def from_config(cls, config_path: str) -> 'TrainerConfig':
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        training_config = conf.get('training', {})
        return cls(
            output_dir=training_config.get('output_dir', 'outputs'),
            batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            num_epochs=training_config.get('num_epochs', 1),
            max_steps=training_config.get('max_steps', 200),
            logging_steps=training_config.get('logging_steps', 1),
            save_steps=training_config.get('save_steps', 50),
            eval_steps=training_config.get('eval_steps'),
            warmup_steps=training_config.get('warmup_steps', 0),
            weight_decay=training_config.get('weight_decay', 0.01),
            seed=training_config.get('seed', 3407)
        )

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class CustomTrainer:
    """Custom trainer for fine-tuning models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config_path: str
    ):
        """Initialize trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = TrainerConfig.from_config(config_path)
        
        self.training_args = self._prepare_training_args()
        self.trainer = None

    def _prepare_training_args(self) -> TrainingArguments:
        """Prepare training arguments.
        
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            seed=self.config.seed,
            report_to="tensorboard"
        )

    def train(self, dataset: Any) -> Dict[str, float]:
        """Train the model.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Dictionary of training metrics
            
        Raises:
            TrainingError: If training fails
        """
        try:
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.model.config.max_position_embeddings,
                dataset_num_proc=2,
                packing=False,
                args=self.training_args
            )
            
            logger.info("Starting training...")
            result = self.trainer.train()
            logger.info("Training completed successfully")
            
            return {
                "train_loss": result.training_loss,
                "train_runtime": result.training_time,
                "train_samples_per_second": result.training_throughput
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise TrainingError(f"Training failed: {e}")

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save the trained model.
        
        Args:
            output_dir: Optional directory to save the model
            
        Raises:
            TrainingError: If saving fails
        """
        try:
            save_path = output_dir or self.config.output_dir
            self.trainer.save_model(save_path)
            logger.info(f"Model saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise TrainingError(f"Failed to save model: {e}")