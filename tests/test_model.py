"""Tests for the model management module."""

import unittest
from unittest.mock import patch, MagicMock
from model import ModelManager, ModelError

class TestModelManager(unittest.TestCase):

    @patch('model.FastLanguageModel')
    def test_load_model(self, mock_fast_language_model):
        # Mock the FastLanguageModel.from_pretrained method
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_fast_language_model.from_pretrained.return_value = (mock_model, mock_tokenizer)

        manager = ModelManager('dummy_config.yaml')
        model, tokenizer = manager.load_model()

        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch('model.FastLanguageModel')
    def test_prepare_for_training(self, mock_fast_language_model):
        # Mock the FastLanguageModel.get_peft_model method
        mock_peft_model = MagicMock()
        mock_fast_language_model.get_peft_model.return_value = mock_peft_model

        manager = ModelManager('dummy_config.yaml')
        manager.model = MagicMock()  # Set a mock model
        prepared_model = manager.prepare_for_training('dummy_config.yaml')

        self.assertEqual(prepared_model, mock_peft_model)

    def test_save_model(self):
        manager = ModelManager('dummy_config.yaml')
        manager.model = MagicMock()
        manager.tokenizer = MagicMock()

        # Mock the save_pretrained_gguf method
        manager.model.save_pretrained_gguf = MagicMock()

        manager.save_model('dummy_output_dir')

        # Assert that save_pretrained_gguf was called
        manager.model.save_pretrained_gguf.assert_called_once()

if __name__ == '__main__':
    unittest.main()