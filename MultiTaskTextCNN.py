#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import sys
import os
os.environ['KERAS_BACKEND']='tensorflow' # Why theano why not

from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D, concatenate
from keras.layers import Concatenate, Lambda, Dense, Input, Flatten, Reshape, Dropout
from keras import regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend import cast
import matplotlib.pyplot as plt
from keras.optimizers import Adam
plt.switch_backend('agg')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


MAX_SEQUENCE_LENGTH = 4000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1


# In[3]:


# reading data
df = pd.read_csv('data/train_including_features.csv')
df = df.dropna()
df = df.reset_index(drop=True)
print('Shape of dataset ',df.shape)
print(df.columns)


# In[4]:


AGE_CLASS_NUM = 7
GENDER_CLASS_NUM = 3
EDU_CLASS_NUM = 7


# In[10]:


x_text = []
age_y = []
gender_y = []
edu_y = []
x_features = []
for index, row in df.iterrows():
    x_features.append([row['space_count'], row['query_count'], row['query_length'], row['how_count'], row['Q_space_ratio'],row['Q_en_ratio']])
    x_text.append(row['query_seg_list'])
    age_y.append(row['Age'])
    gender_y.append(row['Gender'])
    edu_y.append(row['Education'])


# In[11]:


embeddings_index = {}
word_to_idx = {}
idx_to_word = {}
idx = 0

word_to_idx['<PAD>'] = idx
idx_to_word[idx] = '<PAD>'
embeddings_index[idx] = np.random.rand(100)
idx += 1
word_to_idx['<UNK>'] = idx
idx_to_word[idx] = '<UNK>'
embeddings_index[idx] = np.random.rand(100)
idx += 1

f = open('data/w2v/w2v_corpus.vetor',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    idx += 1
    word_to_idx[word] = idx
    word_to_idx[idx] = word
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[idx] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


# In[12]:


x_text_id = []
for text in x_text:
    text = text.replace("\t", " ").split(" ")
    text_id = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text]
    x_text_id.append(text_id)


# In[13]:


x_features_scaled = preprocessing.scale(x_features)
features = x_features_scaled


# In[16]:


MAX_SEQUENCE_LENGTH = 4000
VALIDATION_SPLIT = 0.1
data = pad_sequences(x_text_id, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post", value=word_to_idx['<PAD>'])

age_labels = to_categorical(np.asarray(age_y))
gender_labels = to_categorical(np.asarray(gender_y))
edu_labels = to_categorical(np.asarray(edu_y))


print('Shape of Data Tensor:', data.shape)
print('Shape of Age Label Tensor:', age_labels.shape)
print('Shape of Gender Label Tensor:', gender_labels.shape)
print('Shape of Edu Label Tensor:', edu_labels.shape)
print('Shape of Features:', features.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
age_labels = age_labels[indices]
gender_labels = gender_labels[indices]
edu_labels = edu_labels[indices]
features = features[indices]


# In[17]:


embedding_matrix = np.random.random((len(word_to_idx) + 1, EMBEDDING_DIM))
for word, i in word_to_idx.items():
    embedding_vector = embeddings_index.get(i)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_to_idx) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)


# In[42]:``


def evaluate_model(train_data, val_data):
    x_train, features_train, age_y_train, gender_y_train, edu_y_train = train_data
    x_val, features_val, age_y_val, gender_y_val, edu_y_val = val_data
    
    # define model
    filter_sizes = [2, 3,4, 5]
    num_filters = 100
    drop = 0.1

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="text_input")
    feature_input = Input(shape=(features.shape[1],), dtype="float32",name="feature_input")

    embedded_sequences = embedding_layer(sequence_input)
    reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedded_sequences)
    convolution1 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu')(reshape)
    convolution2 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu')(reshape)
    convolution3 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu')(reshape)
    convolution4 = Conv2D(num_filters, (filter_sizes[3], EMBEDDING_DIM),activation='relu')(reshape)
    maxpooling1 = MaxPooling2D((MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding = 'valid')(convolution1)
    maxpooling2 = MaxPooling2D((MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding = 'valid')(convolution2)
    maxpooling3 = MaxPooling2D((MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding = 'valid')(convolution3)
    maxpooling4 = MaxPooling2D((MAX_SEQUENCE_LENGTH - filter_sizes[3] + 1, 1), strides=(1,1), padding = 'valid')(convolution4)
    merged = concatenate([maxpooling1, maxpooling2, maxpooling3, maxpooling4], axis=1)
    flatten = Flatten()(merged)
    dropout = Dropout(drop)(flatten)

    x = Concatenate(axis=-1)([dropout, feature_input])

    #x_age = Dense(64, activation='relu')(x)
    age_preds = Dense(AGE_CLASS_NUM, activation='softmax', name="age_out")(x)
    #x_gender = Dense(64, activation='relu')(x)
    gender_preds = Dense(GENDER_CLASS_NUM, activation='softmax', name="gender_out")(x)

    #x_edu = Dense(64, activation='relu')(x)
    edu_preds = Dense(EDU_CLASS_NUM, activation='softmax', name="edu_out")(x)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Model([sequence_input, feature_input], [age_preds, gender_preds, edu_preds])
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
    early_stop = EarlyStopping(monitor='val_gender_out_acc', patience = 6)
    cp = ModelCheckpoint('./cross_validation/model_cnn.h5',monitor='val_gender_out_acc',verbose=1,save_best_only=True)
    history=model.fit([x_train, features_train], [age_y_train, gender_y_train, edu_y_train], validation_data=([x_val, features_val], [age_y_val, gender_y_val, edu_y_val]),epochs=30, batch_size=32,callbacks=[cp, early_stop])
    model.load_weights('./cross_validation/model_cnn.h5') 
    val_results = model.evaluate([x_val, features_val], [age_y_val, gender_y_val, edu_y_val], verbose = 1)
    loss, age_out_loss, gender_out_loss, edu_out_loss, age_out_acc, gender_out_acc, edu_out_acc = val_results

    return model, [ age_out_acc, gender_out_acc, edu_out_acc]
    
    


# In[27]:


def split_train_val(data, features, age_labels, gender_labels, edu_labels, test_size):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    age_labels = age_labels[indices]
    gender_labels = gender_labels[indices]
    edu_labels = edu_labels[indices]
    features = features[indices]

    nb_validation_samples = int(test_size * data.shape[0])

    x_train = data[:-nb_validation_samples]
    features_train = features[:-nb_validation_samples]
    age_y_train = age_labels[:-nb_validation_samples]
    gender_y_train = gender_labels[:-nb_validation_samples]
    edu_y_train = edu_labels[:-nb_validation_samples]
    
    x_val = data[-nb_validation_samples:]
    features_val = features[-nb_validation_samples:]
    age_y_val = age_labels[-nb_validation_samples:]
    gender_y_val = gender_labels[-nb_validation_samples:]
    edu_y_val = edu_labels[-nb_validation_samples:]
    
    return [x_train, features_train, age_y_train, gender_y_train, edu_y_train], [x_val, features_val, age_y_val, gender_y_val, edu_y_val]


# In[ ]:

# shuffle data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
age_labels = age_labels[indices]
gender_labels = gender_labels[indices]
edu_labels = edu_labels[indices]
features = features[indices]


n_folds = 5
age_cv_scores, gender_cv_scores, edu_cv_scores, model_history = list(), list(), list(), list()
best = 0
best_age_out_acc, best_gender_out_acc, best_edu_out_acc = 0, 0, 0
best_model = None
rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=2652124)
i = 0
for train_index, test_index in rkf.split(data):
    # split data
    x_train, x_val = data[train_index], data[test_index]
    features_train, features_val = features[train_index], features[test_index]
    age_y_train, age_y_val = age_labels[train_index], age_labels[test_index]
    gender_y_train, gender_y_val = gender_labels[train_index], gender_labels[test_index]
    edu_y_train, edu_y_val = edu_labels[train_index], edu_labels[test_index]
    train_data, val_data = [x_train, features_train, age_y_train, gender_y_train, edu_y_train], [x_val, features_val, age_y_val, gender_y_val, edu_y_val]
    # evaluate model
    model, val_result = evaluate_model(train_data, val_data)
    age_out_acc, gender_out_acc, edu_out_acc = val_result
    if age_out_acc + gender_out_acc + edu_out_acc > best:
        best = age_out_acc + gender_out_acc + edu_out_acc
        best_age_out_acc, best_gender_out_acc, best_edu_out_acc = age_out_acc, gender_out_acc, edu_out_acc 
        best_model = model
    print('fold: %d, age_acc: %.3f, gender_acc: %3f, edu_acc: %3f' % (i, age_out_acc, gender_out_acc, edu_out_acc))
    age_cv_scores.append(age_out_acc)
    gender_cv_scores.append(gender_out_acc)
    edu_cv_scores.append(edu_out_acc)
    model_history.append(model)
print('best, age_acc: %.3f, gender_acc: %3f, edu_acc: %3f' % (best_age_out_acc, best_gender_out_acc, best_edu_out_acc))
    
print('Age Estimated Accuracy %.3f (%.3f), %s' % (np.mean(age_cv_scores), np.std(age_cv_scores), age_cv_scores))
print('Gender Estimated Accuracy %.3f (%.3f), %s' % (np.mean(gender_cv_scores), np.std(gender_cv_scores), gender_cv_scores))
print('Edu Estimated Accuracy %.3f (%.3f), %s' % (np.mean(edu_cv_scores), np.std(edu_cv_scores), edu_cv_scores))
best_model.save("./cross_validation/best_model_cnn.h5")

# In[157]:


def plot_cuv(name, val_name, fig_name, history):
    fig1 = plt.figure()
    plt.plot(history.history[name],'r',linewidth=3.0)
    plt.plot(history.history[val_name],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves :CNN',fontsize=16)
    fig1.savefig(fig_name)
    #plt.show()


# In[ ]:


'''
plot_cuv("age_out_loss", "val_age_out_loss", "age_loss.png", model)
plot_cuv("gender_out_loss", "val_gender_out_loss", "gender_loss.png", model)
plot_cuv("edu_out_loss", "val_edu_out_loss", "edu_loss.png", model)

plot_cuv("age_out_acc", "val_age_out_acc", "age_acc.png", model)
plot_cuv("gender_out_loss", "val_gender_out_loss", "gender_acc.png", model)
plot_cuv("edu_out_loss", "val_edu_out_loss", "edu_acc.png", model)
'''


# In[ ]:


#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)


# In[85]:


#from PIL import Image
#display(Image.open('cnn_model.png'))


# In[ ]:




