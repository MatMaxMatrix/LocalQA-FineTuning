"""Main script for running the LLM fine-tuning process."""

import argparse
import logging
from pathlib import Path

from utils import load_config, setup_logging
from processor import get_processor, ProcessingConfig
from question_generator import QuestionGenerator
from model import ModelManager
from trainer import CustomTrainer

logger = logging.getLogger(__name__)

def main(config_path: str):
    """Run the main fine-tuning process."""
    config = load_config(config_path)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))

    # Process documents
    processing_config = ProcessingConfig.from_config(config_path)
    processor = get_processor(
        config['data']['input_dir'],
        config['data']['file_type'],
        processing_config
    )
    dataset = processor.process()

    # Generate questions
    question_gen = QuestionGenerator(config_path)
    training_data = question_gen.prepare_training_data(dataset)

    # Load and prepare model
    model_manager = ModelManager(config_path)
    model, tokenizer = model_manager.load_model()
    model = model_manager.prepare_for_training(config_path)

    # Train model
    trainer = CustomTrainer(model, tokenizer, config_path)
    metrics = trainer.train(training_data)
    logger.info(f"Training metrics: {metrics}")

    # Save model
    output_dir = Path(config['training']['output_dir'])
    model_manager.save_model(output_dir, config['training']['save_format'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning process")
    parser.add_argument("config", help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)