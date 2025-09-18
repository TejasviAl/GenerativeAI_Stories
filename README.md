**Generative AI Stories Generator**

A Generative AI-based Story Generator built using Hugging Face Transformers and PyTorch.
This project allows you to train a language model on your own dataset of stories and generate unique, creative stories across different themes such as fantasy, sci-fi, horror, romance, and comedy.

**Features**

Train GPT-2 on your own dataset (stories.txt)

Generate 5 unique stories for the same input prompt

Multiple story themes included

Adjustable creativity using temperature & top-p sampling

GPU support for faster training and generation

**Dataset**

Create a file called stories.txt

Each line should be a separate story or paragraph:
      Once upon a time, there was a brave knight...
      A long time ago in a distant kingdom...
      Deep in the enchanted forest lived a fairy...
      
**Fine-tune GPT-2 on your dataset with: ** 
      python train.py
