import unittest
from unittest.mock import patch, MagicMock
from src.data.question_generator import QuestionGenerator
from datasets import Dataset

class TestQuestionGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = QuestionGenerator('configs/config.yaml')

    @patch('src.data.question_generator.requests.post')
    def test_generate_questions(self, mock_post):
        # Mock the response from the Ollama API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '[{"question": "Test question?", "answer": "Test answer."}]'
        }
        mock_post.return_value = mock_response

        # Call the _generate_questions method
        result = self.generator._generate_questions("Sample paragraph.")

        # Assert that the request was made with correct parameters
        mock_post.assert_called_once()
        
        # Check the returned result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['question'], "Test question?")
        self.assertEqual(result[0]['answer'], "Test answer.")

    @patch('src.data.question_generator.QuestionGenerator._generate_questions')
    def test_prepare_training_data(self, mock_generate_questions):
        # Mock the _generate_questions method
        mock_generate_questions.return_value = [
            {"question": "Test question?", "answer": "Test answer."}
        ]

        # Create a dummy dataset
        dummy_dataset = Dataset.from_dict({"text": ["Sample text"]})

        # Call prepare_training_data
        result = self.generator.prepare_training_data(dummy_dataset)

        # Check the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['instruction'], "Test question?")
        self.assertEqual(result[0]['input'], "")
        self.assertEqual(result[0]['output'], "Based on the document: Test answer.")

if __name__ == '__main__':
    unittest.main()