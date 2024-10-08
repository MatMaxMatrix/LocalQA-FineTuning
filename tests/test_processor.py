"""Tests for the document processing module."""

import unittest
from unittest.mock import patch, MagicMock
from processor import WordProcessor, PDFProcessor, ProcessingError, ProcessingConfig

class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.config = ProcessingConfig(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". "])

    @patch('processor.Document')
    def test_word_processor(self, mock_document):
        # Mock the Document class
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Test paragraph 1."), MagicMock(text="Test paragraph 2.")]
        mock_document.return_value = mock_doc

        processor = WordProcessor('dummy_file.docx', self.config)
        result = processor.process()

        # Check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        
        # Check the content of the Dataset
        self.assertEqual(len(result), 1)  # Assuming the text is short enough to fit in one chunk
        self.assertIn("Test paragraph 1. Test paragraph 2.", result[0]['text'])

    @patch('processor.fitz.open')
    def test_pdf_processor(self, mock_fitz_open):
        # Mock the fitz.open method
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test PDF content."
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        processor = PDFProcessor('dummy_file.pdf', self.config)
        result = processor.process()

        # Check that the result is a Dataset
        self.assertIsInstance(result, Dataset)
        
        # Check the content of the Dataset
        self.assertEqual(len(result), 1)  # Assuming the text is short enough to fit in one chunk
        self.assertIn("Test PDF content.", result[0]['text'])

if __name__ == '__main__':
    unittest.main()