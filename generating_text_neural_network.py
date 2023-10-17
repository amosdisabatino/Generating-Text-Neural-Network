import random
import numpy as np
import tensorflow as tf

Sequential = tf.keras.models.Sequential

RMSprop = tf.keras.optimizers.RMSprop

layers = tf.keras.layers
Activation = layers.Activation
Dense = layers.Dense
lstm = layers.LSTM

url = 'https://storage.googleapis.com/download.tensorflow.org/data/'\
      'shakespeare.txt'
filepath = tf.keras.utils.get_file('shakespeare.txt', url)

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))

char_to_index = dict(
    (charact, number) for number, charact in enumerate(characters)
)
index_to_char = dict(
    (number, charact) for number, charact in enumerate(characters)
)

SEQ_LENGHT = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGHT, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGHT])
    next_char.append(text[i + SEQ_LENGHT])

# This array is 3D and it is based on:
# - the amount of all sentences of our dataset,
# - the length of them,
# - the amount of possible characters.
# In this array is stored the information about which character appears at
# which position in which sentences and set the value to True.
x = np.zeros((len(sentences), SEQ_LENGHT, len(characters)), dtype=np.bool_)

# This array is 2D, and the shape is based on:
# - the amount of sentences of our dataset,
# - the amount of possible characters;
# When a character is the next character for a given sentence, we set the
# position to True.
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

# In this for loop:
# 1- We get ecach sentence of the dataset and its index with the `enumerate`
# function;
for num, sentence in enumerate(sentences):
    # For each sentence we get all characters and their index with the
    # `enumerate` function;
    for index, character in enumerate(sentence):
        # We use the `char_to_index` dictionary in order to get the right index
        # for the given character.
        # EX: The character 'g' has gotten on the index 17 in the
        # 'char_to_index' dictionary, if it occurs in the third sentece
        # (index 2) and the fourth position of the sentence (index 3) we would
        # set: x[num, index, char_to_index[character]] = x[2,3,17] to 1
        x[num, index, char_to_index[character]] = 1
    # EX: 't' is in the index 0 of the 'next_char' array, we get them from the
    # array, and from the 'char_to_index' dictionary we get the number of 't'
    # character.
    y[num, char_to_index[next_char[num]]] = 1

"""

model = Sequential()
model.add(lstm(128, input_shape=(SEQ_LENGHT, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)

model.save('generating_text.model')

"""

model = tf.keras.models.load_model('generating_text.model')


def sample(preds, temperature=1.0):
    """
    This method is designed to sample a discrete probability distribution.

    :param: `preds`: This is an array of probabilities. It represents the
    probabilities associated with different options or classes. For example,
    if you were working on a language model, preds could be an array of
    probabilities for different words or characters that might be generated
    later.

    :param: `temperature`: This is an optional parameter that adjusts how
    "random" the choice is. A higher value of temperatures makes the
    probability distribution more uniform, while a lower value makes it more
    focused on the maximum probability. The default value is 1.0, which means a
    balanced choice.

    Now, let's see what the step-by-step method does:

    1 - preds is transformed into an array of type float64 to ensure that
    subsequent mathematical operations are compatible with the floating-point
    numbers.

    2 - The natural logarithm (log) of the probabilities in the preds array is
    calculated and then divided by the temperature value. This step is used to
    adjust how "sparse" or "concentrated" the resulting probability
    distribution will be.

    3 - The exponential of the probabilities obtained from the previous step is
    calculated, resulting in a new array called exp_preds. This step is part of
    the transformation to obtain a smoother probability distribution.

    4 - The exp_preds array is normalized, dividing each value by the sum of
    all values in the array. This ensures that the sum of all probabilities in
    the array is 1, thus creating a valid probability distribution.

    5 - The np.random.multinomial function is used to sample a single value
    from the probability distribution represented by preds. This sampled value
    will be 1 at a position corresponding to one of the possible options and 0
    at the others.

    6 - Finally, the method returns the index of the option with value 1 in the
    sampled array, which corresponds to the choice made.
    """
    # 1
    preds = np.array(preds).astype('float64')
    # 2
    preds = np.log(preds) / temperature
    # 3
    exp_preds = np.exp(preds)
    # 4
    preds = exp_preds / np.sum(exp_preds)
    # 5
    probas = np.random.multinomial(1, preds, 1)
    # 6
    return np.argmax(probas)


def generate_text(lenght, temperature):
    """
    This Python function is designed to generate text using a character-based
    prediction model (a neural network) and a stochastic sampling method to
    creatively generate text.

    1 - It starts by randomly choosing an initial index (start_index)
    within an input text (text) minus the sequence length (SEQ_LENGHT) minus
    one. Presumably, SEQ_LENGHT is a variable defined elsewhere in the code,
    but is not provided in the function itself.

    2 - It starts constructing the generated string as an empty string.

    3- It extracts an initial sequence (sentence) from start_index to
    start_index + SEQ_LENGHT within the input text. This initial sequence will
    be used as the starting point for generating the subsequent text.

    4 - Adds the start sequence to generated with the string " NEW: " to begin
    text generation.

    5 - It enters a loop that iterates length times, generating one character
    at a time to extend the generated text.

    6 - For each iteration in the loop, it prepares the input (x_predictions)
    for the model. This input is a one-hot encoding vector representing the
    current sequence (sentence), so that the model can make predictions based
    on it. The vector is a numpy matrix with dimensions
    (1, SEQ_LENGHT, len(characters)).

    7 - It makes predictions using the model on the vector x_predictions.
    Presumably, the model is a previously trained machine learning model that
    can generate successive characters based on the current sequence.

    8 - It uses a sample function (which is not defined in the function) to
    sample the next character based on the model's predictions and the
    temperature specified as the function argument. The temperature controls
    the randomness of the predictions: higher values make the predictions
    more random, while lower values make them more deterministic.

    9 - Adds the sampled character (next_character) to the generated string.

    10 - Updates the current sequence (sentence) by removing the first
    character and adding next_character to the end. This update of the current
    sequence allows the model to take the context into account while generating
    the next text.

    11 - At the end of the loop, it returns the generated string, which
    contains the generated text.
    """
    # 1-
    start_index = random.randint(0, len(text) - SEQ_LENGHT - 1)
    # 2-
    generated = ''
    # 3-
    sentence = text[start_index: start_index + SEQ_LENGHT]
    # 5-
    generated += sentence + " NEW: "
    for i in range(lenght):
        # 6-
        x_predictions = np.zeros((1, SEQ_LENGHT, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1
        # 7-
        predictions = model.predict(x_predictions, verbose=0)[0]
        # 8-
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        # 9-
        generated += next_character
        # 10-
        sentence = sentence[1:] + next_character
    # 11-
    return generated


print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))
