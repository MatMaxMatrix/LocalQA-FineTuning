# LocalQA-FineTuning

A comprehensive toolkit for fine-tuning Large Language Models (LLMs) on custom datasets with a focus on question-answering tasks.

## Features

- Document processing for Word and PDF files
- Automatic question-answer pair generation from processed documents
- Configurable model loading and preparation using Unsloth
- Fine-tuning implementation using LoRA (Low-Rank Adaptation)
- Customizable training pipeline
- Extensive configuration options via YAML

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MatMaxMatrix/LocalQA-FineTuning.git
   cd LocalQA-FineTuning
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The project uses a YAML configuration file (`configs/config.yaml`) to manage various settings. Key configuration sections include:

- `data`: Settings for document processing
- `model`: Model loading and preparation options
- `training`: Training hyperparameters and output settings
- `ollama`: Configuration for the question generation API

Customize these settings according to your specific use case and requirements.

## Usage

1. Prepare your document dataset:
   Place your Word or PDF documents in the `data/raw` directory.

2. Run the main script:
   ```
   python src/main.py configs/config.yaml
   ```

   This script will:
   - Process the documents
   - Generate question-answer pairs
   - Load and prepare the model
   - Fine-tune the model
   - Save the fine-tuned model

3. The fine-tuned model will be saved in the directory specified by `output_dir` in the configuration file.

## Project Structure

```
llm-finetune-qa/
│
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/         # Place input documents here
│   └── processed/   # Processed data will be stored here
├── src/
│   ├── data/
│   │   ├── processor.py
│   │   └── question_generator.py
│   ├── training/
│   │   ├── model.py
│   │   └── trainer.py
│   ├── main.py
│   └── utils.py
├── tests/
│   ├── test_model.py
│   ├── test_processor.py
│   ├── test_question_generator.py
│   └── test_trainer.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Testing

To run the unit tests:

```
python -m unittest discover tests
```

## Contributing

Contributions to improve LocalQA-FineTuning are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the [Unsloth](https://github.com/unslothai/unsloth) library for efficient model handling.
- Question generation is powered by the Ollama API.
- We use the [Transformers](https://github.com/huggingface/transformers) library by Hugging Face.

## Disclaimer

This tool is designed for research and educational purposes. Users are responsible for ensuring they have the necessary rights and permissions for any documents they process or content they generate using this toolkit. Always respect copyright and intellectual property rights when using this software.

## Contact

Mobin - azimipanah.mobin@gmail.com

Project Link: [https://github.com/MatMaxMatrix/LocalQA-FineTuning](https://github.com/MatMaxMatrix/LocalQA-FineTuning)