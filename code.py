!pip install nltk # using nltk for tokenization

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk

# Dataset
document = """In the heart of Africa, the vast Sahara Desert stretches endlessly, its golden sands whispering tales of ancient caravans that once traversed its dunes.
From the bustling streets of Cairo, Egypt, to the serene landscapes of Cape Town, South Africa, the continent pulses with life and history.

To the north, Europe beckons with its rich tapestry of cultures. In Paris, France, the Eiffel Tower stands as a testament to architectural brilliance, while the canals of Venice, Italy, offer a glimpse into a romantic past.
The historic city of Prague, Czech Republic, with its cobblestone streets, echoes the footsteps of poets and philosophers.

Across the Atlantic, the Americas unfold their own stories.
In the United States, the skyscrapers of New York City reach for the heavens, while the beaches of Rio de Janeiro, Brazil, invite relaxation and festivity.
The Andes Mountains, stretching through countries like Peru and Chile, harbor ancient civilizations and breathtaking vistas.

Asia, the largest continent, is a mosaic of contrasts.
Tokyo, Japan, hums with technological advancements, while Kyoto preserves the tranquility of traditional tea ceremonies.
In India, the Taj Mahal stands as a symbol of eternal love, and the bustling markets of Delhi offer a sensory overload of colors and sounds.

Oceania, often overlooked, offers paradisiacal landscapes.
The Great Barrier Reef off the coast of Australia teems with marine life, and New Zealand's rolling hills have served as the backdrop to epic tales.

Each continent, each country, each city has its own unique rhythm and story.
From the icy expanse of Antarctica, where scientists brave the elements to study climate change, to the vibrant festivals of Mexico City, the world is a patchwork of experiences waiting to be explored.

As travelers journey from the fjords of Norway to the savannas of Kenya, they carry with them stories of the places they've been and the people they've met.
These tales, passed down through generations, weave a rich tapestry of human experience, connecting us all through the shared love of discovery and adventure."""

# tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# tokenize
tokens = word_tokenize(document.lower())

# Create a vocabulary from the tokenized words
# Vocabulary is a group of all the unique tokens

vocab = {'<UNK>' : 0} # Reserve token 0 for unknown words

for token in Counter(tokens).keys(): # Extract all unique words/tokens in the dataset
  if token not in vocab: # Add the token to the vocabulary if it doesn't already exist, assigning it a unique index
    vocab[token] = len(vocab) # Add the token to the vocabulary and assign it a unique index based on current length

vocab # Display the final vocabulary dictionary containing all unique tokens and their corresponding indices

len(vocab)  # Returns the total count of unique words in the vocabulary

# Split the document (dataset) into individual sentences
input_sentences = document.split('\n')

# Convert each word in the sentence to its number from the vocabulary
def text_to_indices(sentence, vocab):
  numerical_sentence = []

  for token in sentence: # iterate over every sentence
    if token in vocab: # if the token is in the vocab
      numerical_sentence.append(vocab[token]) # Append the token's corresponding index from the vocabulary to the numerical sentence
    else: # if the token is not in the vocabulary
      numerical_sentence.append(vocab['<UNK>']) # If the word is unknown, use the <UNK> token’s number

  return numerical_sentence

# Tokenize each sentence and convert it to lowercase
input_numerical_sentences = [] # Create a list to store the numerical representation of each input sentence

# Convert each sentence into a numerical sequence using the vocabulary
for sentence in input_sentences:
  input_numerical_sentences.append(text_to_indices(word_tokenize(sentence.lower()), vocab)) # Convert each sentence into a list of numbers using the vocabulary and store them


len( input_numerical_sentences) # the number of sentences we have in the dataset

# Generate training sequences from each sentence in the dataset
# Create incremental training sequences from each sentence
# Each sequence includes the first token up to the current token,
# which is useful for training a next-word prediction model

training_sequence = []  # Initialize a list to store all training sequences generated from the dataset

for sentence in input_numerical_sentences: # iterate over each numerical sentence to create training sequences

  for i in range(1, len(sentence)): # generate all sub-sequences (parts) of the current sentence -> starting from the second word (index 1) to the last word
    training_sequence.append(sentence[:i+1]) # Append the current sub-sequence (from the first token to the i-th token) to the training_sequence list

training_sequence[:5] # Display the first 5 generated training sequences

# find out the longest sequence in the training_sequence list
len_list = [] # Initialize an empty list to store lengths of sequences in training_sequence

for sequence in training_sequence:
  len_list.append(len(sequence)) # store the length of every sequence

max(len_list) # get the length of the longest sequence in training_sequence

# Padding all training sequences to the same length

padded_training_sequence = [] # initialize an empty list to store the zero-padded training sequences

# iterate through each sequence in the training data to apply zero padding
for sequence in training_sequence:

  # make a list of 0's for each sequence
  padded_training_sequence.append([0]*(max(len_list) - len(sequence)) + sequence) # add the required number of 0's at the beginning of each sequence to match the maximum sequence length

len(padded_training_sequence[12]) # All sequences are padded to match the length of the longest sequence

 # convert the 2D list (padded_training_sequence) to tensor
padded_training_sequence = torch.tensor(padded_training_sequence, dtype=torch.long)

padded_training_sequence # display the final padded training sequence tensor

padded_training_sequence.shape # 378 rows, 45 columns

# Separate input (X) and output (y) from the tensor 'padded_training_sequence'
# In each row, all elements except the last one represent the input (X)
# The last element in each row represents the target output (y)

X = padded_training_sequence[:, :-1] # x contains all elements of each row except the last one (input features)
y = padded_training_sequence[:,-1] # output value (target) is the last element of each sequence / row

# Dataset class
# Create the dataset class
class CustomDataset(Dataset): # CustomDataset inherits from PyTorch's Dataset class

  # Constructor method to initialize input features (X) and labels (y)
  def __init__(self, X, y):
    self.X = X
    self.y = y

  # Return the number of samples in the dataset
  def __len__(self):
    return self.X.shape[0]

  # Retrieve the features (X) and label (y) at the given index
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
# Dataset object
# 'dataset' is an object of the 'CustomDataset' class
dataset = CustomDataset(X, y) # Create an instance of the CustomDataset class using input features X and labels y

len(dataset) # length of the dataset

# first sentence - input feature (X) & true labels / output (y)
dataset[0]

# DataLoader object
# 'dataloader' is an object of the 'DataLoader' class
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # Initialize a DataLoader for the dataset with batch size 32 and shuffling enabled

# Iterate through the DataLoader to access each batch of inputs and outputs
for input, output in dataloader:
  print(input, output)

# Build the LSTM architecture
# The 'LSTMModel' class inherits from PyTorch's 'nn.Module' base class
class LSTMModel(nn.Module):

  # constructor method
  def __init__(self, vocab_size):
    super().__init__() # Initialize the parent class
    self.embedding = nn.Embedding(vocab_size, 100) # Converts each token index to a 100-dimensional embedding vector
    self.lstm = nn.LSTM(100, 150, batch_first=True) # LSTM layer with input size 100 (from embeddings) and outputs hidden states of size 150
    self.fc = nn.Linear(150, vocab_size) # Fully connected (linear) layer: takes 150 features from the LSTM output as input and produces logits for each word in the vocabulary (one logit per word)

  # Defines the forward pass
  def forward(self, X):
    embedded_input = self.embedding(X) # Pass the input tensor through the embedding layer
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded_input) # Pass the embedded input through the LSTM layer
                                                                                                   # In an LSTM, h_t represents the hidden state and c_t represents the cell state at time step t
    output = self.fc(final_hidden_state.squeeze(0)) # Pass the last time step's hidden state (h_t) through the fully connected layer to get the output
    return output # Return the final prediction from the fully connected layer

# Define and build the LSTM-based neural network model

# Create an instance of the LSTMModel using the vocabulary size
model = LSTMModel(len(vocab))

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Important variables
epochs = 50
learning_rate = 0.001

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
# training loop
for epoch in range(epochs):

  # to calculate the total loss
  total_loss = 0

  # Iterate over each batch of input data (batch_X) and labels (batch_y)
  for batch_X, batch_y in dataloader:

    # move input data (batch_X) & labels (batch_y) to GPU
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    # clear gradients
    optimizer.zero_grad()

    # forward pass
    output = model(batch_X) # Pass input data through the model to get predictions

    # Compute the loss between model predictions and true labels
    loss = criterion(output, batch_y)

    # Perform backpropagation to compute gradients
    loss.backward()

    # Update model parameters (weights and biases)
    optimizer.step()

    # Accumulate the current loss into the total loss
    total_loss = total_loss + loss.item()

  # Print the current epoch number and total loss (formatted to 4 decimal places)
  print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")

# Perform the prediction
# prediction

# Makes a prediction using the trained model for the given input text.
def prediction(model, vocab, text):

  # Convert the text to lowercase and split it into individual words (tokens)
  tokenized_text = word_tokenize(text.lower())

  # Convert tokenized text into numerical index representation
  numerical_text = text_to_indices(tokenized_text, vocab)

  # Add zero-padding to the start of the numerical sequence so all sequences have equal length
  padded_text = torch.tensor([0] * (max(len_list) - len(numerical_text)) + numerical_text, dtype=torch.long).unsqueeze(0)

  # ensure same device as model
  device = next(model.parameters()).device # get the device model is on
  padded_text = padded_text.to(device) # move input tensor to same device

  # Pass the padded input text to the model to obtain the output
  output = model(padded_text)

 # Get the index of the word with the highest predicted probability
  value, index = torch.max(output, dim=1)

  # Retrieve the word corresponding to the given index
  # Append the retrieved word to the existing text and return the updated text
  return text + " " + list(vocab.keys())[index]



# Generate words step by step using the prediction model and show each word with a delay

# Example 1
# Import the 'time' module to use functions like sleep for pausing execution
import time

num_tokens = 10 # Define how many tokens (words) to generate in total
input_text = "The Great Barrier Reef off the coast of" # Initialize the starting text prompt

# Loop 'num_tokens' times to generate each new token iteratively
for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text) # Call the 'prediction' function to predict the next word(s) based on the model and vocabulary
  print(output_text) # Print the generated text after each prediction
  input_text = output_text # Update 'input_text' so the next prediction uses the latest output as context
  time.sleep(0.5) # Pause the program for 0.5 seconds before generating the next token

# Example 2
# Import the 'time' module to use functions like sleep for pausing execution
import time

num_tokens = 10 # Define how many tokens (words) to generate in total
input_text = "In India, the Taj Mahal" # Initialize the starting text prompt

# Loop 'num_tokens' times to generate each new token iteratively
for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text) # Call the 'prediction' function to predict the next word(s) based on the model and vocabulary
  print(output_text) # Print the generated text after each prediction
  input_text = output_text # Update 'input_text' so the next prediction uses the latest output as context
  time.sleep(0.5) # Pause the program for 0.5 seconds before generating the next token

# Example 3
# Import the 'time' module to use functions like sleep for pausing execution
import time

num_tokens = 10 # Define how many tokens (words) to generate in total
input_text = "In Paris, France," # Initialize the starting text prompt

# Loop 'num_tokens' times to generate each new token iteratively
for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text) # Call the 'prediction' function to predict the next word(s) based on the model and vocabulary
  print(output_text) # Print the generated text after each prediction
  input_text = output_text # Update 'input_text' so the next prediction uses the latest output as context
  time.sleep(0.5) # Pause the program for 0.5 seconds before generating the next token

# Example 4
# Import the 'time' module to use functions like sleep for pausing execution
import time

num_tokens = 10 # Define how many tokens (words) to generate in total
input_text = "In the United States, the skyscrapers of New York City" # Initialize the starting text prompt

# Loop 'num_tokens' times to generate each new token iteratively
for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text) # Call the 'prediction' function to predict the next word(s) based on the model and vocabulary
  print(output_text) # Print the generated text after each prediction
  input_text = output_text # Update 'input_text' so the next prediction uses the latest output as context
  time.sleep(0.5) # Pause the program for 0.5 seconds before generating the next token

# Example 5
# Import the 'time' module to use functions like sleep for pausing execution
import time

num_tokens = 10 # Define how many tokens (words) to generate in total
input_text = "In the heart of Africa," # Initialize the starting text prompt

# Loop 'num_tokens' times to generate each new token iteratively
for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text) # Call the 'prediction' function to predict the next word(s) based on the model and vocabulary
  print(output_text) # Print the generated text after each prediction
  input_text = output_text # Update 'input_text' so the next prediction uses the latest output as context
  time.sleep(0.5) # Pause the program for 0.5 seconds before generating the next token