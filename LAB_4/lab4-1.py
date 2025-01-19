import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# 1. Завантаження та обробка датасету
file_path = './ukr.txt'  # Вкажіть шлях до датасету
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

english_sentences = []
ukrainian_sentences = []
for line in lines:
    parts = line.split('\t')
    if len(parts) >= 2:
        english_sentences.append(parts[0].strip())
        ukrainian_sentences.append(parts[1].strip())

# Додаємо спеціальні токени
start_token = '<s>'
end_token = '<e>'
ukrainian_sentences = [f"{start_token} {sentence} {end_token}" for sentence in ukrainian_sentences]

# Лімітуємо кількість даних
data_limit = 10000  # Збільшено обсяг даних
english_sentences = english_sentences[:data_limit]
ukrainian_sentences = ukrainian_sentences[:data_limit]

# Токенізація тексту
MAX_NUM_WORDS = 30000  # Збільшено розмір словника
MAX_SEQ_LENGTH = 40  # Збільшено довжину послідовностей

tokenizer_eng = Tokenizer(num_words=MAX_NUM_WORDS, filters='', lower=True)
tokenizer_ukr = Tokenizer(num_words=MAX_NUM_WORDS, filters='', lower=True)

tokenizer_eng.fit_on_texts(english_sentences)
tokenizer_ukr.fit_on_texts(ukrainian_sentences)

input_sequences = tokenizer_eng.texts_to_sequences(english_sentences)
target_sequences = tokenizer_ukr.texts_to_sequences(ukrainian_sentences)

# Паддінг
encoder_input_data = pad_sequences(input_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

# Визначаємо словники
vocab_size_eng = len(tokenizer_eng.word_index) + 1
vocab_size_ukr = len(tokenizer_ukr.word_index) + 1

# 2. Побудова моделі трансформера
EMBEDDING_DIM = 256  # Збільшено розмір ембеддингів
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 6  # Збільшено кількість шарів


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model

    def call(self, x):
        angle_rads = self.get_angles(
            np.arange(self.position)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
            self.d_model
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return x + tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates


# Модель трансформера

def create_transformer_model(vocab_size_enc, vocab_size_dec, seq_length):
    encoder_inputs = Input(shape=(seq_length,), name="encoder_inputs")
    decoder_inputs = Input(shape=(seq_length,), name="decoder_inputs")

    # Ембеддинг та позиційне кодування
    embedding_layer_enc = Embedding(vocab_size_enc, EMBEDDING_DIM)(encoder_inputs)
    embedding_layer_dec = Embedding(vocab_size_dec, EMBEDDING_DIM)(decoder_inputs)

    positional_encoding_enc = PositionalEncoding(seq_length, EMBEDDING_DIM)(embedding_layer_enc)
    positional_encoding_dec = PositionalEncoding(seq_length, EMBEDDING_DIM)(embedding_layer_dec)

    # Transformer layers (спрощений)
    transformer_output = Dense(EMBEDDING_DIM, activation="relu")(positional_encoding_enc)
    transformer_output = Dropout(0.2)(transformer_output)  # Збільшено Dropout
    transformer_output = LayerNormalization(epsilon=1e-6)(transformer_output)

    outputs = Dense(vocab_size_dec, activation="softmax")(transformer_output)

    return Model([encoder_inputs, decoder_inputs], outputs)


model = create_transformer_model(vocab_size_eng, vocab_size_ukr, MAX_SEQ_LENGTH)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 3. Навчання
BATCH_SIZE = 64
EPOCHS = 20  # Збільшено кількість епох

history = model.fit(
    [encoder_input_data, pad_sequences(decoder_input_data[:, :-1], maxlen=MAX_SEQ_LENGTH, padding='post')],
    pad_sequences(decoder_input_data[:, 1:], maxlen=MAX_SEQ_LENGTH, padding='post'),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# 4. Інференс
print("Навчання завершено. Модель готова до тестування!")


def beam_search_translate(input_sentence, beam_width=3):
    input_seq = tokenizer_eng.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=MAX_SEQ_LENGTH, padding='post')

    sequences = [[list(), 0.0]]  # Список послідовностей з їхніми логарифмічними ймовірностями

    for _ in range(MAX_SEQ_LENGTH):
        all_candidates = []
        for seq, score in sequences:
            decoder_seq = np.zeros((1, MAX_SEQ_LENGTH))
            for t, token in enumerate(seq):
                decoder_seq[0, t] = token

            prediction = model.predict([input_seq, decoder_seq])
            top_k = np.argsort(prediction[0, len(seq) - 1])[-beam_width:]
            for word_id in top_k:
                candidate = [seq + [word_id], score - np.log(prediction[0, len(seq) - 1, word_id])]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

        if all(seq[-1] == tokenizer_ukr.word_index[end_token] for seq, _ in sequences):
            break

    best_sequence = sequences[0][0]
    translated_sentence = ' '.join(tokenizer_ukr.index_word.get(word_id, '<unknown>') for word_id in best_sequence if
                                   word_id > 0 and word_id != tokenizer_ukr.word_index[end_token])
    return translated_sentence


# Багаторазовий переклад вручну
while True:
    user_input = input("Введіть англійське речення для перекладу (або 'exit' для виходу): ")
    if user_input.lower() == 'exit':
        print("Завершення програми.")
        break
    print("Translated:", beam_search_translate(user_input))