🧠 Next Word Predictor using LSTM

This project implements an LSTM-based Next Word Prediction model that generates coherent and context-aware text continuations. The model is trained on a custom world narrative dataset describing diverse countries, cities, and cultures. Built using PyTorch and NLTK, the model learns sequential language patterns and predicts the next word based on prior context.

🚀 Project Overview

1. The Next Word Predictor demonstrates the power of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, for sequence modeling tasks in Natural Language Processing (NLP).
2. It processes a raw text dataset, tokenizes it, converts words into numerical sequences, and trains a model to predict the next token in a sentence. The system can generate new, grammatically consistent sentences when provided with an initial input prompt.

🧩 Key Features

🔤 Text Tokenization & Vocabulary Building — Implemented using NLTK, creating unique token indices for efficient sequence modeling.

🧱 Custom PyTorch Dataset & DataLoader — Enables seamless batching, padding, and efficient GPU-based training.

🧮 LSTM-based Neural Network — Learns temporal dependencies to predict the next word in a sequence.

🧠 Contextual Text Generation — Produces realistic and semantically relevant sentence continuations.

📉 Optimized Training Pipeline — Achieved a final training loss of 1.3494 after 50 epochs.

🧠 Model Architecture
| Layer                     | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| **Embedding Layer**       | Converts token indices into 100-dimensional dense vectors        |
| **LSTM Layer**            | Learns sequential dependencies with hidden size = 150            |
| **Fully Connected Layer** | Maps hidden states to vocabulary logits for next-word prediction |

🧰 Technologies Used

Python | PyTorch | NLTK | NumPy | Matplotlib

🧪 Training Details

Epochs: 50

Optimizer: Adam

Learning Rate: 0.001

Loss Function: CrossEntropyLoss

Final Training Loss: 1.3494

📊 Dataset

The model is trained on a custom world narrative dataset describing locations, cultures, and experiences across continents — including Africa, Europe, Asia, the Americas, and Oceania.
It’s designed to teach the model contextual transitions between words and regions.

💡 Example Predictions

Input:
"In the heart of Africa,"

Output:
"In the heart of Africa, the vast Sahara Desert stretches endlessly..."

Input:
"In Paris, France,"

Output:
"In Paris, France, the Eiffel Tower stands as a testament to architectural brilliance..."
