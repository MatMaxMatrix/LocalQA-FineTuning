"""Tests for the training module."""

import unittest
from unittest.mock import patch, MagicMock
from trainer import CustomTrainer, TrainingError

class TestCustomTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.trainer = CustomTrainer(self.mock_model, self.mock_tokenizer, 'dummy_config.yaml')

    @patch('trainer.SFTTrainer')
    def test_train(self, mock_sft_trainer):
        # Mock the SFTTrainer
        mock_trainer = MagicMock()
        mock_sft_trainer.return_value = mock_trainer

        # Mock the train method result
        mock_result = MagicMock()
        mock_result.training_loss = 0.5
        mock_result.training_time = 100
        mock_result.training_throughput = 10
        mock_trainer.train.return_value = mock_result

        # Create a dummy dataset
        dummy_dataset = [{"text": "Sample text"}]

        # Call the train method
        metrics = self.trainer.train(dummy_dataset)

        # Assert that SFTTrainer was called with correct arguments
        mock_sft_trainer.assert_called_once()

        # Assert that train was called
        mock_trainer.train.assert_called_once()

        # Check the returned metrics
        self.assertEqual(metrics['train_loss'], 0.5)
        self.assertEqual(metrics['train_runtime'], 100)
        self.assertEqual(metrics['train_samples_per_second'], 10)

    def test_save_model(self):
        # Mock the trainer
        self.trainer.trainer = MagicMock()

        # Call save_model
        self.trainer.save_model('dummy_output_dir')

        # Assert that save_model was called on the trainer
        self.trainer.trainer.save_model.assert_called_once_with('dummy_output_dir')

if __name__ == '__main__':
    unittest.main()