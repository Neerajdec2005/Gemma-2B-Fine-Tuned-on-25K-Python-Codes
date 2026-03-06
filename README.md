# Fine-Tuning Gemma 2B for Python Code Generation

## Overview

This project demonstrates fine-tuning Google's Gemma 2B model for code generation tasks using Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation). The implementation is provided as a Jupyter notebook designed for Google Colab, leveraging quantization techniques to optimize memory usage.

## Features

- **Quantized Model Loading**: Uses 4-bit quantization (NF4) with BitsAndBytes for efficient memory usage
- **LoRA Fine-Tuning**: Applies Low-Rank Adaptation to target modules for efficient training
- **Supervised Fine-Tuning**: Utilizes TRL's SFTTrainer for instruction-based fine-tuning
- **Code Generation**: Fine-tunes the model on Python code generation tasks
- **Google Colab Integration**: Seamlessly runs in Google Colab environment with Hugging Face authentication

## Prerequisites

- Google Colab account with GPU access (recommended)
- Hugging Face account with access to Gemma models
- Basic understanding of Python and machine learning concepts

## Installation

1. Open the `Gemma_Finetuning.ipynb` notebook in Google Colab
2. Ensure GPU runtime is selected (Runtime > Change runtime type > Hardware accelerator > GPU)
3. Create a Hugging Face token with read access to gated models
4. Add your token to Colab secrets as `Token`

## Usage

1. **Setup Environment**:
   - Run the first cell to install required dependencies
   - Import necessary libraries and set up authentication

2. **Load Model**:
   - Configure BitsAndBytes quantization
   - Load Gemma-2B model with quantization

3. **Test Base Model**:
   - Generate sample text to verify model loading

4. **Configure Fine-Tuning**:
   - Set up LoRA configuration
   - Load and prepare the dataset

5. **Train Model**:
   - Initialize SFTTrainer with training arguments
   - Run training for specified steps

6. **Evaluate Results**:
   - Test the fine-tuned model on code generation tasks

## Dataset

The project uses the `flytech/python-codes-25k` dataset from Hugging Face, which contains:
- Python code snippets
- Corresponding instruction queries
- 25,000 training examples

The dataset is formatted to include both the query and expected code output for supervised fine-tuning.

## Training Configuration

- **Batch Size**: 1 per device with gradient accumulation of 4
- **Learning Rate**: 2e-4
- **Optimizer**: Paged AdamW 8-bit
- **Max Steps**: 100 (configurable)
- **LoRA Rank**: 8
- **Target Modules**: Query, Output, Key, Value, Gate, Up, and Down projections

## Results

After fine-tuning, the model demonstrates improved capability in generating Python code based on natural language queries. Note that with only 100 training steps, results may be limited; increasing to 500-1000 steps typically yields better performance.

## Dependencies

- `bitsandbytes`: For quantization
- `peft`: Parameter-Efficient Fine-Tuning
- `trl`: Transformer Reinforcement Learning
- `accelerate`: Distributed training
- `datasets`: Hugging Face datasets
- `transformers`: Hugging Face transformers

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is provided as-is for educational and research purposes. Please refer to the licenses of individual dependencies and the Gemma model usage terms.

## Acknowledgments

- Google for the Gemma model
- Hugging Face for the transformers library and datasets
- The open-source community for PEFT and TRL implementations
- Special thanks to Krishna Naik for inspiration and guidance in fine-tuning techniques