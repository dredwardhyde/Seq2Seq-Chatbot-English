import codecs
import io
import os
import re
import zipfile

import numpy as np
import requests
from gensim.models import Word2Vec
from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

########################################################################################################################
########################################### DATA PREPARATION ###########################################################
########################################################################################################################
r = requests.get('http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()


def get_all_conversations():
    all_conversations = []
    with codecs.open("./cornell movie-dialogs corpus/movie_lines.txt", "rb", encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\n")
        for line in lines:
            all_conversations.append(line.split(" +++$+++ "))
    return all_conversations


def get_all_sorted_chats(all_conversations):
    all_chats = {}
    # get only first 10000 conversations from dataset because whole dataset will take 9.16 TiB of RAM
    for tokens in all_conversations[:10000]:
        if len(tokens) > 4:
            all_chats[int(tokens[0][1:])] = tokens[4]
    return sorted(all_chats.items(), key=lambda x: x[0])


def clean_text(text_to_clean):
    res = text_to_clean.lower()
    res = re.sub(r"i'm", "i am", res)
    res = re.sub(r"he's", "he is", res)
    res = re.sub(r"she's", "she is", res)
    res = re.sub(r"it's", "it is", res)
    res = re.sub(r"that's", "that is", res)
    res = re.sub(r"what's", "that is", res)
    res = re.sub(r"where's", "where is", res)
    res = re.sub(r"how's", "how is", res)
    res = re.sub(r"\'ll", " will", res)
    res = re.sub(r"\'ve", " have", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"\'d", " would", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"won't", "will not", res)
    res = re.sub(r"can't", "cannot", res)
    res = re.sub(r"n't", " not", res)
    res = re.sub(r"n'", "ng", res)
    res = re.sub(r"'bout", "about", res)
    res = re.sub(r"'til", "until", res)
    res = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", res)
    return res


def get_conversation_dict(sorted_chats):
    conversations_dictionary = {}
    counter = 1
    conversations_ids = []
    for i in range(1, len(sorted_chats) + 1):
        if i < len(sorted_chats):
            if (sorted_chats[i][0] - sorted_chats[i - 1][0]) == 1:
                if sorted_chats[i - 1][1] not in conversations_ids:
                    conversations_ids.append(sorted_chats[i - 1][1])
                conversations_ids.append(sorted_chats[i][1])
            elif (sorted_chats[i][0] - sorted_chats[i - 1][0]) > 1:
                conversations_dictionary[counter] = conversations_ids
                conversations_ids = []
            counter += 1
        else:
            continue
    return conversations_dictionary


def get_clean_q_and_a(conversations_dictionary):
    context_and_target = []
    for current_conversations in conversations_dictionary.values():
        if len(current_conversations) % 2 != 0:
            current_conversations = current_conversations[:-1]
        for i in range(0, len(current_conversations), 2):
            context_and_target.append((current_conversations[i], current_conversations[i + 1]))
    context, target = zip(*context_and_target)
    context_dirty = list(context)
    clean_questions = list()
    for i in range(len(context_dirty)):
        clean_questions.append(clean_text(context_dirty[i]))
    target_dirty = list(target)
    clean_answers = list()
    for i in range(len(target_dirty)):
        clean_answers.append('<START> ' + clean_text(target_dirty[i]) + ' <END>')
    return clean_questions, clean_answers


conversations = get_all_conversations()
print("Total conversations in dataset: {}".format(len(conversations)))
all_sorted_chats = get_all_sorted_chats(conversations)
conversation_dictionary = get_conversation_dict(all_sorted_chats)
questions, answers = get_clean_q_and_a(conversation_dictionary)

########################################################################################################################
############################################# MODEL TRAINING ###########################################################
########################################################################################################################
print(len(questions))
print(len(answers))
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789')
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

print('VOCAB SIZE : {}'.format(VOCAB_SIZE))

vocab = []
for word in tokenizer.word_index:
    vocab.append(word)


def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary


p = tokenize(questions + answers)
model = Word2Vec(p[0], min_count=1)

embedding_matrix = np.zeros((VOCAB_SIZE, 100))
for i in range(len(tokenizer.word_index)):
    embedding_matrix[i] = model[vocab[i]]

tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
padded_questions = pad_sequences(tokenized_questions, maxlen=maxlen_questions,
                                 padding='post')
encoder_input_data = np.array(padded_questions)

print(encoder_input_data.shape, maxlen_questions)

tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
padded_answers = pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(padded_answers)

print(decoder_input_data.shape, maxlen_answers)

tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
onehot_answers = to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(onehot_answers)

print(decoder_output_data.shape)

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation=softmax)
output = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=50, epochs=300)
model.save('model_big.h5')


# model.load_weights('model_big.h5')


def make_inference_models():
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(200,))
    decoder_state_input_c = Input(shape=(200,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


enc_model, dec_model = make_inference_models()

for _ in range(100):
    states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)
