import numpy as np
import pandas as pd
import re
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Concatenate,BatchNormalization,Activation
from keras.layers import Lambda, Embedding, GRU, Bidirectional, TimeDistributed, concatenate
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from word2vecReader import Word2Vec
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
import pickle
K.set_learning_phase(1)
np.random.seed(0)
MAX_SENT_LENGTH = 20###number of words in a sentence
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 400
POST_DIM=30
VALIDATION_SPLIT = 0.2

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

def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
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
        self.b = K.variable(self.init((self.attention_dim, )))
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
texts=dictionary['text']
#texts=[text.encode('ascii') for text in texts]
comments=dictionary['comments']
timeInfo=dictionary['time']
postInfo=dictionary['post']
labels=dictionary['labels']

b = np.zeros([len(timeInfo),len(max(timeInfo,key = lambda x: len(x)))])
for i,j in enumerate(timeInfo):
    b[i][0:len(j)] = j
timeInfo=b
time_size=len(np.unique(timeInfo))
MAX_SENTS = len(timeInfo[0])####number of sentences

postInfo=np.array(postInfo)
post_size=len(np.unique(postInfo))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

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
single_label=np.asarray(labels)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

###standardize
timeInfo = preprocessing.StandardScaler().fit_transform(timeInfo)
postInfo = preprocessing.StandardScaler().fit_transform(postInfo)
HAN_pre = []
HAN_reca = []
HAN_f1 = []
HAN_AUC = []
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

for j in range(10):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data1 = data[indices]
    print(data1.shape)
    labels1 = labels[indices]
    single_label1=single_label[indices]
    timeInfo1=timeInfo[indices]
    print (timeInfo1.shape)
    timeInfo1=timeInfo1.reshape((2218,MAX_SENTS,1))
    print (timeInfo1.shape)
    data1=np.dstack((data1, timeInfo1))
    postInfo1=postInfo[indices]
    print (postInfo1.shape)
    nb_validation_samples = int(VALIDATION_SPLIT * data1.shape[0])
    zeros=np.zeros(2218)
    zeros=zeros.reshape((2218,1,1))

    x_train = data1[:-nb_validation_samples]
    y_train = labels1[:-nb_validation_samples]
    zeros_train=zeros[:-nb_validation_samples]
    time_train=timeInfo1[:-nb_validation_samples]
    post_train=postInfo1[:-nb_validation_samples]
    x_val = data1[-nb_validation_samples:]
    y_val = labels1[-nb_validation_samples:]
    zeros_test=zeros[-nb_validation_samples:]
    time_test=timeInfo1[-nb_validation_samples:]
    post_test=postInfo1[-nb_validation_samples:]
    y_single=single_label1[-nb_validation_samples:]

    print('Number of positive and negative posts in training and test set')
    print (y_train.sum(axis=0))
    print (y_val.sum(axis=0))

    # building Hierachical Attention network

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True)

    all_input = Input(shape=(MAX_SENT_LENGTH+1,))
    sentence_input=crop(1, 0, MAX_SENT_LENGTH)(all_input)##slice
    time_input=crop(1, MAX_SENT_LENGTH, MAX_SENT_LENGTH+1)(all_input)##slice
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)####(?,200)
    #time_embedding=Dense(TIME_DIM,activation='sigmoid')(time_input)
    merged_output=Concatenate()([l_att,time_input])###text+time information
    sentEncoder = Model(all_input, merged_output)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH+1))
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    #pred_time=Dense(1,activation='relu')(l_lstm_sent)
    fully_sent=Dense(1,use_bias=False)(l_lstm_sent)
    norm_fullysent=BatchNormalization()(fully_sent)
    pred_time=Activation(activation='linear')(norm_fullysent)

    zero_input=Input(shape=(1,1))
    shift_predtime=Concatenate(axis=1)([zero_input,pred_time])
    shift_predtime=crop(1, 0, MAX_SENTS)(shift_predtime)
    l_att_sent = AttLayer(100)(l_lstm_sent)

    ###embed the #likes, shares
    post_input=Input(shape=(4,))
    #post_embedding = Dense(POST_DIM, activation='sigmoid')(post_input)
    fully_post=Dense(POST_DIM,use_bias=False)(post_input)
    norm_fullypost=BatchNormalization()(fully_post)
    post_embedding=Activation(activation='relu')(norm_fullypost)
    x = concatenate([l_att_sent,post_embedding])###merge the document level vectro with the additional embedded features such as #likes
    fully_review=Dense(2,use_bias=False)(x)
    norm_fullyreview=BatchNormalization()(fully_review)
    preds=Activation(activation='softmax')(norm_fullyreview)

    rmsprop = optimizers.adam(lr=0.001)
    model = Model(inputs=[review_input,post_input,zero_input], outputs=[preds,shift_predtime])
    #print(model.summary())
    model.compile(loss=['binary_crossentropy','mse'],loss_weights=[1,0.00002],
                  optimizer=rmsprop)
    #filepath = "weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [EarlyStopping(monitor='loss', patience=1,mode='min'),checkpoint]

    print("model fitting - Hierachical attention network for cyberbullying detection")

    model.fit([x_train,post_train,zeros_train], [y_train,time_train], batch_size=16,
              epochs=1,verbose=1)
    yp = model.predict([x_val,post_test,zeros_test], verbose=1)
    ypreds=yp[0]
    ypreds = np.argmax(ypreds, axis=1)
    #print y_single
    #print ypred
    f1=precision_recall_fscore_support(y_single, ypreds)
    auc=roc_auc_score(y_single, ypreds)
    print(f1)
    print(auc)
    HAN_AUC.append(auc)
    HAN_f1.append(f1[2][1])
    HAN_reca.append(f1[1][1])
    HAN_pre.append(f1[0][1])

    #for t-sne visualization
    if j==0:
        a=model.layers
        get_representations_test = K.function([model.layers[0].input,model.layers[1].input,model.layers[12].input], [model.layers[6].output])
        representations_test = get_representations_test([x_val,post_test,zeros_test])[0]
        representation_dict = {
            'representations': representations_test,
            'labels': y_single
        }

        with open('HANCD_Tem_results.pickle', 'wb') as handle:
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

