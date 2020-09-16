import numpy as np
import pandas as pd
import re
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Concatenate, BatchNormalization, Activation, Add
from keras.layers import Lambda, Embedding, GRU, Bidirectional, TimeDistributed, concatenate
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from word2vecReader import Word2Vec
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import math
import pickle
K.set_learning_phase(1)
#np.random.seed(0)
MAX_SENT_LENGTH = 20  ###number of words in a sentence
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 400
POST_DIM = 30
VALIDATION_SPLIT = 0.2
d_model = 100


##slice tensor function in keras
def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)


def myFunc(x):
    if "empety" in x:
        return False
    else:
        return True


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = string.strip().lower()
    word_tokens = word_tokenize(string)
    filtered_words = [word for word in word_tokens if word not in stopwords.words('english')]
    return filtered_words


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index + len(char)] == char:
                    return index

            index += 1


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

with open('instagram.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
handle.close()
TIME_DIM = 60
HAN_pre=[]
HAN_reca=[]
HAN_f1=[]
HAN_AUC=[]

texts = dictionary['text']
comments = dictionary['comments']
timeInfo_unsort = dictionary['time']
postInfo = dictionary['post']
labels = dictionary['labels']
timeInfo = [sorted(t) for t in timeInfo_unsort]
MAX_SENTS = 150
b = np.zeros([len(timeInfo), MAX_SENTS])
for i, j in enumerate(timeInfo):
    b[i, 0:len(j)] = j[:MAX_SENTS]
timeInfo = b
# timeInfo = timeInfo / 60  # seconds to minutes
time_size = len(np.unique(timeInfo))
# MAX_SENTS = len(timeInfo[0])  ####number of sentences

postInfo = np.array(postInfo)
post_size = len(np.unique(postInfo))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

# lr_list=[1e-5,1e-4,5e-4,1e-3,5e-3,1e-2,1e-1]
lr =1e-3
data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(comments):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
single_label = np.asarray(labels)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

###standardize
# timeInfo = preprocessing.StandardScaler().fit_transform(timeInfo)
postInfo = preprocessing.StandardScaler().fit_transform(postInfo)

embeddings_index = Word2Vec.load_word2vec_format("word2vec_twitter_model.bin", binary=True)  #

# print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
outword_dic = dict()
for word, i in word_index.items():
    if word in embeddings_index.vocab:
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    else:
        new_vector = np.random.rand(EMBEDDING_DIM, )
        outword_dic.setdefault(word, new_vector)
        embedding_matrix[i] = outword_dic[word]
for i in range(20):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    print(data.shape)
    N = data.shape[0]
    texts = [texts[i] for i in indices.tolist()]
    comments = [comments[i] for i in indices.tolist()]
    labels = labels[indices]
    single_label = single_label[indices]
    timeInfo = timeInfo[indices]
    print(timeInfo.shape)
    postInfo = postInfo[indices]
    print(postInfo.shape)
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    all_time = np.repeat(timeInfo[:, 0:MAX_SENTS, np.newaxis], d_model / 2, axis=2)
    div_term = np.exp(np.arange(0, d_model, 2).astype(float) * (-math.log(100.0) / d_model))
    pe = np.zeros((2218, MAX_SENTS, d_model))
    pe[:, :, 0::2] = np.sin(all_time * div_term)
    pe[:, :, 1::2] = np.cos(all_time * div_term)

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    pe_train = pe[:-nb_validation_samples]
    post_train = postInfo[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    post_test = postInfo[-nb_validation_samples:]
    pe_test = pe[-nb_validation_samples:]
    y_single = single_label[-nb_validation_samples:]

    print('Number of positive and negative posts in training and validation set')
    print (y_train.sum(axis=0))
    print (y_val.sum(axis=0))

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,))
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)  ####(?,200)
    # calculate positional encoding
    # time_embed = Input(shape=(d_model,))
    # position embedding + word embedding
    # all_output=Concatenate()([l_att,time_embed])
    sentEncoder = Model([sentence_input], l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH))
    time_input = Input(shape=(MAX_SENTS, d_model))
    # concat=Concatenate()([review_input,time_input])
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    time_embed = Dense(TIME_DIM, use_bias=False)(time_input)
    norm_time = BatchNormalization()(time_embed)
    all_output = Concatenate()([review_encoder, norm_time])
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(all_output)
    l_att_sent = AttLayer(100)(l_lstm_sent)

    ###embed the #likes, shares
    post_input = Input(shape=(4,))
    # post_embedding = Dense(POST_DIM, activation='sigmoid')(post_input)
    fully_post = Dense(POST_DIM, use_bias=False)(post_input)
    norm_fullypost = BatchNormalization()(fully_post)
    post_embedding = Activation(activation='relu')(norm_fullypost)
    x = concatenate([l_att_sent,
                     post_embedding])  ###merge the document level vectro with the additional embedded features such as #likes
    fully_review = Dense(2, use_bias=False)(x)
    norm_fullyreview = BatchNormalization()(fully_review)
    preds = Activation(activation='softmax')(norm_fullyreview)

    rmsprop = optimizers.adam(lr=lr)
    model = Model(inputs=[review_input, post_input, time_input], outputs=preds)
    #print(model.summary())
    model.compile(loss=['binary_crossentropy'],
                  optimizer=rmsprop)
    # filepath = "weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [EarlyStopping(monitor='loss', patience=1,mode='min'),checkpoint]

    print("model fitting - Hierachical attention network for cyberbullying detection")

    model.fit([x_train, post_train, pe_train], [y_train], validation_data=([x_val, post_test, pe_test], [y_val]), batch_size=16,
              epochs=1, verbose=1)
    yp = model.predict([x_val, post_test, pe_test], verbose=1)
    ypreds = np.argmax(yp, axis=1)

    f1 = precision_recall_fscore_support(y_single, ypreds)
    auc = roc_auc_score(y_single, ypreds)
    print(f1)
    print(auc)
    HAN_AUC.append(auc)
    HAN_f1.append(f1[2][1])
    HAN_reca.append(f1[1][1])
    HAN_pre.append(f1[0][1])

    #for t-sne visualization
    if i ==0:
        a = model.layers
        get_representations_test = K.function([model.layers[1].input,model.layers[5].input,model.layers[0].input], [model.layers[10].output])
        representations_test = get_representations_test([x_val, post_test, pe_test])[0]
        representation_dict = {
            'representations': representations_test,
            'labels': y_single
        }

        with open('Positionresults.pickle', 'wb') as handle:
            pickle.dump(representation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    K.clear_session()

print HAN_AUC
print HAN_f1
print HAN_pre
print HAN_reca
print ("AUC",np.mean(HAN_AUC), np.std(HAN_AUC))
print ("f1", np.mean(HAN_f1), np.std(HAN_f1))
print ("precision",np.mean(HAN_pre), np.std(HAN_pre))
print ("recall", np.mean(HAN_reca), np.std(HAN_reca))

