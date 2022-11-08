import scanpy as sc
import scipy.io
import scprep
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
import random
import os

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Convolution1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import sys


adata = pd.read_csv('./myData_pancreatic_5batches.txt',sep='\t',header=0, index_col=0)
adata = sc.AnnData(np.transpose(adata))
d4_genes = list(adata.var_names)
sample_adata = pd.read_csv('./mySample_pancreatic_5batches.txt',header=0, index_col=0, sep='\t')
adata.obs['cell_type'] = sample_adata.loc[adata.obs_names,['celltype']]
adata.obs['batch'] = sample_adata.loc[adata.obs_names,['batchlb']]


batch1 = adata[adata.obs["batch"] == "Baron_b1", :]
adata_0 = batch1.X
label_0 = adata.obs['cell_type'][adata.obs['batch']=='Baron_b1']
label_0 = label_0.to_list()
batch2 = adata[adata.obs["batch"] == "Mutaro_b2", :]
adata_1 = batch2.X
label_1 = adata.obs['cell_type'][adata.obs['batch']=='Mutaro_b2']
label_1 = label_1.to_list()
batch3 = adata[adata.obs["batch"] == "Segerstolpe_b3", :]
adata_2 = batch3.X
label_2 = adata.obs['cell_type'][adata.obs['batch']=='Segerstolpe_b3']
label_2 = label_2.to_list()
batch4 = adata[adata.obs["batch"] == "Wang_b4", :]
adata_3 = batch4.X
label_3 = adata.obs['cell_type'][adata.obs['batch']=='Wang_b4']
label_3 = label_3.to_list()
batch5 = adata[adata.obs["batch"] == "Xin_b5", :]
adata_4 = batch5.X
label_4 = adata.obs['cell_type'][adata.obs['batch']=='Xin_b5']
label_4 = label_4.to_list()

label_0_set = set(label_0)
label_1_set = set(label_1)
label_2_set = set(label_2)
label_3_set = set(label_3)
label_4_set = set(label_4)
all_list = list(label_0_set.union(label_1_set,label_2_set,label_3_set,label_4_set))
sub_list = list(label_0_set&label_2_set&label_1_set&label_3_set&label_4_set)

data_ln = scprep.normalize.library_size_normalize(adata_0)
adata_0_pre = scprep.transform.log(data_ln, pseudocount=1, base=10)
data_ln = scprep.normalize.library_size_normalize(adata_1)
adata_1_pre = scprep.transform.log(data_ln, pseudocount=1, base=10)
data_ln = scprep.normalize.library_size_normalize(adata_2)
adata_2_pre = scprep.transform.log(data_ln, pseudocount=1, base=10)
data_ln = scprep.normalize.library_size_normalize(adata_3)
adata_3_pre = scprep.transform.log(data_ln, pseudocount=1, base=10)
data_ln = scprep.normalize.library_size_normalize(adata_4)
adata_4_pre = scprep.transform.log(data_ln, pseudocount=1, base=10)

pca = TruncatedSVD(50)
adata_0_pca = pca.fit_transform(adata_0_pre)
adata_1_pca = pca.fit_transform(adata_1_pre)
adata_2_pca = pca.fit_transform(adata_2_pre)
adata_3_pca = pca.fit_transform(adata_3_pre)
adata_4_pca = pca.fit_transform(adata_4_pre)


def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def Create_Model():
    img_rows = 50
    nb_filters = 32
    pool_size = 2
    kernel_size = 3
    input_shape = (img_rows, 1)

    model = Sequential()

    model.add(Convolution1D(64, 5, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution1D(64, 5, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Convolution1D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution1D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution1D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution1D(nb_filters, 1, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Convolution1D(nb_filters, 1, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Convolution1D(nb_filters, 1, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    return model


def euclidean_distance(vects):
    eps = 1e-5
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(model):
    nb_classes = len(all_list)

    epoch = 15  # Epoch number
    batch_size = 512

    print('Training the model - Epoch ' + str(epoch))
    nn = batch_size
    best_Acc = 0
    for e in range(epoch):

        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(len(y2) // nn):  ##Align batch 1 and batch 2
            loss = model.train_on_batch([X12[i * nn:(i + 1) * nn, :], X2[i * nn:(i + 1) * nn, :]],
                                        [y12[i * nn:(i + 1) * nn, ], yc12[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X2[i * nn:(i + 1) * nn, :], X12[i * nn:(i + 1) * nn, :]],
                                        [y2[i * nn:(i + 1) * nn, ], yc12[i * nn:(i + 1) * nn, ]])
        for i in range(len(y3) // nn):  ##Align batch 1 and batch 3
            loss = model.train_on_batch([X13[i * nn:(i + 1) * nn, :], X3[i * nn:(i + 1) * nn, :]],
                                        [y13[i * nn:(i + 1) * nn, ], yc13[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X3[i * nn:(i + 1) * nn, :], X13[i * nn:(i + 1) * nn, :]],
                                        [y3[i * nn:(i + 1) * nn, ], yc13[i * nn:(i + 1) * nn, ]])
        for i in range(len(y4) // nn):  ##Align batch 1 and batch 4
            loss = model.train_on_batch([X14[i * nn:(i + 1) * nn, :], X4[i * nn:(i + 1) * nn, :]],
                                        [y14[i * nn:(i + 1) * nn, ], yc14[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X4[i * nn:(i + 1) * nn, :], X14[i * nn:(i + 1) * nn, :]],
                                        [y4[i * nn:(i + 1) * nn, ], yc14[i * nn:(i + 1) * nn, ]])
        for i in range(len(y5) // nn):  ##Align batch 1 and batch 5
            loss = model.train_on_batch([X15[i * nn:(i + 1) * nn, :], X5[i * nn:(i + 1) * nn, :]],
                                        [y15[i * nn:(i + 1) * nn, ], yc15[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X5[i * nn:(i + 1) * nn, :], X15[i * nn:(i + 1) * nn, :]],
                                        [y5[i * nn:(i + 1) * nn, ], yc15[i * nn:(i + 1) * nn, ]])

        for i in range(len(y3_2) // nn):  ##Align batch 2 and batch 3
            loss = model.train_on_batch([X23[i * nn:(i + 1) * nn, :], X3_2[i * nn:(i + 1) * nn, :]],
                                        [y23[i * nn:(i + 1) * nn, ], yc23[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X3_2[i * nn:(i + 1) * nn, :], X23[i * nn:(i + 1) * nn, :]],
                                        [y3_2[i * nn:(i + 1) * nn, ], yc23[i * nn:(i + 1) * nn, ]])
        for i in range(len(y4_2) // nn):  ##Align batch 2 and batch 4
            loss = model.train_on_batch([X24[i * nn:(i + 1) * nn, :], X4_2[i * nn:(i + 1) * nn, :]],
                                        [y24[i * nn:(i + 1) * nn, ], yc24[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X4_2[i * nn:(i + 1) * nn, :], X24[i * nn:(i + 1) * nn, :]],
                                        [y4_2[i * nn:(i + 1) * nn, ], yc24[i * nn:(i + 1) * nn, ]])
        for i in range(len(y5_2) // nn):  ##Align batch 2 and batch 5
            loss = model.train_on_batch([X25[i * nn:(i + 1) * nn, :], X5_2[i * nn:(i + 1) * nn, :]],
                                        [y25[i * nn:(i + 1) * nn, ], yc25[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X5_2[i * nn:(i + 1) * nn, :], X25[i * nn:(i + 1) * nn, :]],
                                        [y5_2[i * nn:(i + 1) * nn, ], yc25[i * nn:(i + 1) * nn, ]])
        for i in range(len(y4_3) // nn):  ##Align batch 3 and batch 4
            loss = model.train_on_batch([X34[i * nn:(i + 1) * nn, :], X4_3[i * nn:(i + 1) * nn, :]],
                                        [y34[i * nn:(i + 1) * nn, ], yc34[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X4_3[i * nn:(i + 1) * nn, :], X34[i * nn:(i + 1) * nn, :]],
                                        [y4_3[i * nn:(i + 1) * nn, ], yc34[i * nn:(i + 1) * nn, ]])
        for i in range(len(y5_3) // nn):  ##Align batch 3 and batch 5
            loss = model.train_on_batch([X35[i * nn:(i + 1) * nn, :], X5_3[i * nn:(i + 1) * nn, :]],
                                        [y35[i * nn:(i + 1) * nn, ], yc35[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X5_3[i * nn:(i + 1) * nn, :], X35[i * nn:(i + 1) * nn, :]],
                                        [y5_3[i * nn:(i + 1) * nn, ], yc35[i * nn:(i + 1) * nn, ]])
        for i in range(len(y5_4) // nn):  ##Align batch 4 and batch 5
            loss = model.train_on_batch([X45[i * nn:(i + 1) * nn, :], X5_4[i * nn:(i + 1) * nn, :]],
                                        [y45[i * nn:(i + 1) * nn, ], yc45[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X5_4[i * nn:(i + 1) * nn, :], X45[i * nn:(i + 1) * nn, :]],
                                        [y5_4[i * nn:(i + 1) * nn, ], yc45[i * nn:(i + 1) * nn, ]])
            ##monitor the classification accuracy
        Out_0 = model.predict([X_test_0, X_test_0])
        Acc_v_0 = np.argmax(Out_0[0], axis=1) - np.argmax(y_test_0, axis=1)

        Acc_0 = (len(Acc_v_0) - np.count_nonzero(Acc_v_0) + .0000001) / len(Acc_v_0)

        Out_1 = model.predict([X_test_1, X_test_1])
        Acc_v_1 = np.argmax(Out_1[0], axis=1) - np.argmax(y_test_1, axis=1)

        Acc_1 = (len(Acc_v_1) - np.count_nonzero(Acc_v_1) + .0000001) / len(Acc_v_1)

        Out_2 = model.predict([X_test_2, X_test_2])
        Acc_v_2 = np.argmax(Out_2[0], axis=1) - np.argmax(y_test_2, axis=1)

        Acc_2 = (len(Acc_v_2) - np.count_nonzero(Acc_v_2) + .0000001) / len(Acc_v_2)

        Out_3 = model.predict([X_test_3, X_test_3])
        Acc_v_3 = np.argmax(Out_3[0], axis=1) - np.argmax(y_test_3, axis=1)

        Acc_3 = (len(Acc_v_3) - np.count_nonzero(Acc_v_3) + .0000001) / len(Acc_v_3)

        Out_4 = model.predict([X_test_4, X_test_4])
        Acc_v_4 = np.argmax(Out_4[0], axis=1) - np.argmax(y_test_4, axis=1)

        Acc_4 = (len(Acc_v_4) - np.count_nonzero(Acc_v_4) + .0000001) / len(Acc_v_4)

        print(
            'acc0=' + str(Acc_0) + 'acc1=' + str(Acc_1) + ' acc2=' + str(Acc_2) + ' acc3=' + str(Acc_3) + 'acc4=' + str(
                Acc_4) + 'loss=' + str(loss))
    print(str(e))
    return best_Acc

label_0 = np.array(label_0)
label_1 = np.array(label_1)
label_2 = np.array(label_2)
label_3 = np.array(label_3)
label_4 = np.array(label_4)

l0_0 = list(np.where(label_0=='acinar')[0])
l1_0 = list(np.where(label_0=='alpha')[0])
l2_0 = list(np.where(label_0=='beta')[0])
l3_0 = list(np.where(label_0=='delta')[0])
l4_0 = list(np.where(label_0=='ductal')[0])
l5_0 = list(np.where(label_0=='endothelial')[0])
l6_0 = list(np.where(label_0=='epsilon')[0])
l7_0 = list(np.where(label_0=='gamma')[0])
l8_0 = list(np.where(label_0=='macrophage')[0])
l9_0 = list(np.where(label_0=='mast')[0])
l10_0 = list(np.where(label_0=='schwann')[0])
l11_0 = list(np.where(label_0=='stellate')[0])
l12_0 = list(np.where(label_0=='t_cell')[0])

l0_1 = list(np.where(label_1=='acinar')[0])
l1_1 = list(np.where(label_1=='alpha')[0])
l2_1 = list(np.where(label_1=='beta')[0])
l3_1 = list(np.where(label_1=='delta')[0])
l4_1 = list(np.where(label_1=='ductal')[0])
l5_1 = list(np.where(label_1=='endothelial')[0])
l6_1 = list(np.where(label_1=='epsilon')[0])
l7_1 = list(np.where(label_1=='gamma')[0])
l8_1 = list(np.where(label_1=='mesenchymal')[0])

l0_2 = list(np.where(label_2=='MHC class II')[0])
l1_2 = list(np.where(label_2=='acinar')[0])
l2_2 = list(np.where(label_2=='alpha')[0])
l3_2 = list(np.where(label_2=='beta')[0])
l4_2 = list(np.where(label_2=='delta')[0])
l5_2 = list(np.where(label_2=='ductal')[0])
l6_2 = list(np.where(label_2=='endothelial')[0])
l7_2 = list(np.where(label_2=='epsilon')[0])
l8_2 = list(np.where(label_2=='gamma')[0])
l9_2 = list(np.where(label_2=='mast')[0])
l10_2 = list(np.where(label_2=='stellate')[0])

l0_3 = list(np.where(label_3=='acinar')[0])
l1_3 = list(np.where(label_3=='alpha')[0])
l2_3 = list(np.where(label_3=='beta')[0])
l3_3 = list(np.where(label_3=='delta')[0])
l4_3 = list(np.where(label_3=='ductal')[0])
l5_3 = list(np.where(label_3=='gamma')[0])
l6_3 = list(np.where(label_3=='mesenchymal')[0])

l0_4 = list(np.where(label_4=='alpha')[0])
l1_4 = list(np.where(label_4=='beta')[0])
l2_4 = list(np.where(label_4=='delta')[0])
l3_4 = list(np.where(label_4=='gamma')[0])

cell_list_0 = [l0_0,l1_0,l2_0,l3_0,l4_0,l5_0,l6_0,l7_0,l8_0,l9_0,l10_0,l11_0,l12_0]
cell_list_1 = [l0_1,l1_1,l2_1,l3_1,l4_1,l5_1,l6_1,l7_1,l8_1]
cell_list_2 = [l0_2,l1_2,l2_2,l3_2,l4_2,l5_2,l6_2,l7_2,l8_2,l9_2,l10_2]
cell_list_3 = [l0_3,l1_3,l2_3,l3_3,l4_3,l5_3,l6_3]
cell_list_4 = [l0_4,l1_4,l2_4,l3_4]

sample_num = 400
sample_list_0 = []
for i in range(13):
  if len(cell_list_0[i])>=400:
    sample_list_0 = sample_list_0+random.sample(cell_list_0[i],sample_num)
  else:
    sample_list_0 = sample_list_0+cell_list_0[i]

X_train_0 = adata_0_pca[sample_list_0,:]
y_train_0 = np.array(label_0)[sample_list_0]

X0_test = adata_0_pca
y0_test = np.array(label_0)

sample_num = 400
sample_list_1 = []
for i in range(9):
  if len(cell_list_1[i])>=400:
    sample_list_1 = sample_list_1+random.sample(cell_list_1[i],sample_num)
  else:
    sample_list_1 = sample_list_1+cell_list_1[i]

X_train_1 = adata_1_pca[sample_list_1,:]
y_train_1 = np.array(label_1)[sample_list_1]

X1_test = adata_1_pca
y1_test = np.array(label_1)

sample_num = 400
sample_list_2 = []
for i in range(11):
  if len(cell_list_2[i])>=400:
    sample_list_2 = sample_list_2+random.sample(cell_list_2[i],sample_num)
  else:
    sample_list_2 = sample_list_2+cell_list_2[i]

X_train_2 = adata_2_pca[sample_list_2,:]
y_train_2 = np.array(label_2)[sample_list_2]

X2_test = adata_2_pca
y2_test = np.array(label_2)

sample_num = 400
sample_list_3 = []
for i in range(7):
  if len(cell_list_3[i])>=400:
    sample_list_3 = sample_list_3+random.sample(cell_list_3[i],sample_num)
  else:
    sample_list_3 = sample_list_3+cell_list_3[i]

X_train_3 = adata_3_pca[sample_list_3,:]
y_train_3 = np.array(label_3)[sample_list_3]

X3_test = adata_3_pca
y3_test = np.array(label_3)

sample_num = 400
sample_list_4 = []
for i in range(4):
  if len(cell_list_4[i])>=400:
    sample_list_4 = sample_list_4+random.sample(cell_list_4[i],sample_num)
  else:
    sample_list_4 = sample_list_4+cell_list_4[i]

X_train_4 = adata_4_pca[sample_list_4,:]
y_train_4 = np.array(label_4)[sample_list_4]

X4_test = adata_4_pca
y4_test = np.array(label_4)

y_train_1=list(y_train_1)
y_train_2=list(y_train_2)
y_train_0=list(y_train_0)
y_train_3=list(y_train_3)
y_train_4=list(y_train_4)
y0_test = list(y0_test)
y1_test = list(y1_test)
y2_test = list(y2_test)
y3_test = list(y3_test)
y4_test = list(y4_test)

all_list = ['stellate',
 'macrophage',
 'gamma',
 'MHC class II',
 'delta',
 'epsilon',
 'mesenchymal',
 'endothelial',
 'ductal',
 'acinar',
 'mast',
 't_cell',
 'alpha',
 'beta',
 'schwann']

label_set_list = all_list
print(len(label_set_list))
label_code = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

for i in range(len(y_train_0)):
  for j in range(len(label_set_list)):
    if y_train_0[i] == label_set_list[j]:
      y_train_0[i] = label_code[j]
for i in range(len(y_train_1)):
  for j in range(len(label_set_list)):
    if y_train_1[i] == label_set_list[j]:
      y_train_1[i] = label_code[j]
for i in range(len(y_train_2)):
  for j in range(len(label_set_list)):
    if y_train_2[i] == label_set_list[j]:
      y_train_2[i] = label_code[j]
for i in range(len(y_train_3)):
  for j in range(len(label_set_list)):
    if y_train_3[i] == label_set_list[j]:
      y_train_3[i] = label_code[j]
for i in range(len(y_train_4)):
  for j in range(len(label_set_list)):
    if y_train_4[i] == label_set_list[j]:
      y_train_4[i] = label_code[j]

for i in range(len(y0_test)):
    for j in range(len(label_set_list)):
        if y0_test[i] == label_set_list[j]:
            y0_test[i] = label_code[j]
for i in range(len(y1_test)):
    for j in range(len(label_set_list)):
        if y1_test[i] == label_set_list[j]:
            y1_test[i] = label_code[j]
for i in range(len(y2_test)):
    for j in range(len(label_set_list)):
        if y2_test[i] == label_set_list[j]:
            y2_test[i] = label_code[j]
for i in range(len(y3_test)):
    for j in range(len(label_set_list)):
        if y3_test[i] == label_set_list[j]:
            y3_test[i] = label_code[j]
for i in range(len(y4_test)):
    for j in range(len(label_set_list)):
        if y4_test[i] == label_set_list[j]:
            y4_test[i] = label_code[j]


## Pairs for 1&2
Training_P12=[]
Training_N12=[]


for tr0 in range(len(y_train_0)):
    for tr1 in range(len(y_train_1)):

        if y_train_0[tr0]==y_train_1[tr1]:
            Training_P12.append([tr0,tr1])
        else:
            Training_N12.append([tr0,tr1])


random.shuffle(Training_N12)
Training12 = Training_P12+Training_N12[:2*len(Training_P12)]
random.shuffle(Training12)

X12 = np.zeros([len(Training12), 50])
X2 = np.zeros([len(Training12), 50])

y12 = np.zeros([len(Training12)])
y2 = np.zeros([len(Training12)])

yc12 = np.zeros([len(Training12)])

for i in range(len(Training12)):
    in1, in2 = Training12[i]
    X12[i, :] = X_train_0[in1, :]
    X2[i, :] = X_train_1[in2, :]

    y12[i] = y_train_0[in1]
    y2[i] = y_train_1[in2]
    if y_train_0[in1] == y_train_1[in2]:
        yc12[i] = 1

del Training_N12,Training_P12,Training12
import gc
gc.collect()

# Pairs for 1&3

Training_P13 = []
Training_N13 = []

for tr0 in range(len(y_train_0)):
    for tr2 in range(len(y_train_2)):

        if y_train_0[tr0] == y_train_2[tr2]:
            Training_P13.append([tr0, tr2])
        else:
            Training_N13.append([tr0, tr2])

random.shuffle(Training_N13)
Training13 = Training_P13 + Training_N13[:2 * len(Training_P13)]
random.shuffle(Training13)

X13=np.zeros([len(Training13),50])
X3=np.zeros([len(Training13),50])


y13=np.zeros([len(Training13)])
y3=np.zeros([len(Training13)])

yc13=np.zeros([len(Training13)])

for i in range(len(Training13)):
    in1,in3=Training13[i]
    X13[i,:]=X_train_0[in1,:]
    X3[i,:]=X_train_2[in3,:]

    y13[i]=y_train_0[in1]
    y3[i]=y_train_2[in3]
    if y_train_0[in1]==y_train_2[in3]:
        yc13[i]=1
del Training_N13,Training_P13,Training13
gc.collect()

# Pairs for 1&4

Training_P14 = []
Training_N14 = []

for tr0 in range(len(y_train_0)):
    for tr3 in range(len(y_train_3)):

        if y_train_0[tr0] == y_train_3[tr3]:
            Training_P14.append([tr0, tr3])
        else:
            Training_N14.append([tr0, tr3])

random.shuffle(Training_N14)
Training14 = Training_P14 + Training_N14[:2 * len(Training_P14)]
random.shuffle(Training14)

X14=np.zeros([len(Training14),50])
X4=np.zeros([len(Training14),50])


y14=np.zeros([len(Training14)])
y4=np.zeros([len(Training14)])

yc14=np.zeros([len(Training14)])

for i in range(len(Training14)):
    in1,in4=Training14[i]
    X14[i,:]=X_train_0[in1,:]
    X4[i,:]=X_train_3[in4,:]

    y14[i]=y_train_0[in1]
    y4[i]=y_train_3[in4]
    if y_train_0[in1]==y_train_3[in4]:
        yc14[i]=1
del Training_N14,Training_P14,Training14
gc.collect()

# Pairs for 1&5

Training_P15 = []
Training_N15 = []

for tr0 in range(len(y_train_0)):
    for tr4 in range(len(y_train_4)):

        if y_train_0[tr0] == y_train_4[tr4]:
            Training_P15.append([tr0, tr4])
        else:
            Training_N15.append([tr0, tr4])

random.shuffle(Training_N15)
Training15 = Training_P15 + Training_N15[:2 * len(Training_P15)]
random.shuffle(Training15)
X15=np.zeros([len(Training15),50])
X5=np.zeros([len(Training15),50])


y15=np.zeros([len(Training15)])
y5=np.zeros([len(Training15)])

yc15=np.zeros([len(Training15)])

for i in range(len(Training15)):
    in1,in5=Training15[i]
    X15[i,:]=X_train_0[in1,:]
    X5[i,:]=X_train_4[in5,:]

    y15[i]=y_train_0[in1]
    y5[i]=y_train_4[in5]
    if y_train_0[in1]==y_train_4[in5]:
        yc15[i]=1
del Training_N15,Training_P15,Training15
gc.collect()

# Pairs for 2&3

Training_P23 = []
Training_N23 = []

for tr1 in range(len(y_train_1)):
    for tr2 in range(len(y_train_2)):

        if y_train_1[tr1] == y_train_2[tr2]:
            Training_P23.append([tr1, tr2])
        else:
            Training_N23.append([tr1, tr2])

random.shuffle(Training_N23)
Training23 = Training_P23 + Training_N23[:2 * len(Training_P23)]
random.shuffle(Training23)

X23=np.zeros([len(Training23),50])
X3_2=np.zeros([len(Training23),50])


y23=np.zeros([len(Training23)])
y3_2=np.zeros([len(Training23)])

yc23=np.zeros([len(Training23)])

for i in range(len(Training23)):
    in2,in3=Training23[i]
    X23[i,:]=X_train_1[in2,:]
    X3_2[i,:]=X_train_2[in3,:]

    y23[i]=y_train_1[in2]
    y3_2[i]=y_train_2[in3]
    if y_train_1[in2]==y_train_2[in3]:
        yc23[i]=1
del Training_N23,Training_P23,Training23
gc.collect()

# Pairs for 2&4

Training_P24 = []
Training_N24 = []

for tr1 in range(len(y_train_1)):
    for tr3 in range(len(y_train_3)):

        if y_train_1[tr1] == y_train_3[tr3]:
            Training_P24.append([tr1, tr3])
        else:
            Training_N24.append([tr1, tr3])

random.shuffle(Training_N24)
Training24 = Training_P24 + Training_N24[:2 * len(Training_P24)]
random.shuffle(Training24)

X24=np.zeros([len(Training24),50])
X4_2=np.zeros([len(Training24),50])


y24=np.zeros([len(Training24)])
y4_2=np.zeros([len(Training24)])

yc24=np.zeros([len(Training24)])

for i in range(len(Training24)):
    in2,in4=Training24[i]
    X24[i,:]=X_train_1[in2,:]
    X4_2[i,:]=X_train_3[in4,:]

    y24[i]=y_train_1[in2]
    y4_2[i]=y_train_3[in4]
    if y_train_1[in2]==y_train_3[in4]:
        yc24[i]=1

del Training_N24,Training_P24,Training24
gc.collect()

# Pairs for 2&5

Training_P25 = []
Training_N25 = []

for tr1 in range(len(y_train_1)):
    for tr4 in range(len(y_train_4)):

        if y_train_1[tr1] == y_train_4[tr4]:
            Training_P25.append([tr1, tr4])
        else:
            Training_N25.append([tr1, tr4])

random.shuffle(Training_N25)
Training25 = Training_P25 + Training_N25[:2 * len(Training_P25)]
random.shuffle(Training25)
X25=np.zeros([len(Training25),50])
X5_2=np.zeros([len(Training25),50])


y25=np.zeros([len(Training25)])
y5_2=np.zeros([len(Training25)])

yc25=np.zeros([len(Training25)])

for i in range(len(Training25)):
    in2,in5=Training25[i]
    X25[i,:]=X_train_1[in2,:]
    X5_2[i,:]=X_train_4[in5,:]

    y25[i]=y_train_1[in2]
    y5_2[i]=y_train_4[in5]
    if y_train_1[in2]==y_train_4[in5]:
        yc25[i]=1
del Training_N25,Training_P25,Training25
gc.collect()
Training_P34=[]
Training_N34=[]


for tr2 in range(len(y_train_2)):
    for tr3 in range(len(y_train_3)):

        if y_train_2[tr2]==y_train_3[tr3]:
           Training_P34.append([tr2,tr3])
        else:
           Training_N34.append([tr2,tr3])


random.shuffle(Training_N34)
Training34 = Training_P34+Training_N34[:6*len(Training_P34)]
random.shuffle(Training34)
X34=np.zeros([len(Training34),50])
X4_3=np.zeros([len(Training34),50])


y34=np.zeros([len(Training34)])
y4_3=np.zeros([len(Training34)])

yc34=np.zeros([len(Training34)])

for i in range(len(Training34)):
    in3,in4=Training34[i]
    X34[i,:]=X_train_2[in3,:]
    X4_3[i,:]=X_train_3[in4,:]

    y34[i]=y_train_2[in3]
    y4_3[i]=y_train_3[in4]
    if y_train_2[in3]==y_train_3[in4]:
        yc34[i]=1

del Training_N34,Training_P34,Training34
gc.collect()

Training_P35=[]
Training_N35=[]


for tr2 in range(len(y_train_2)):
    for tr4 in range(len(y_train_4)):

        if y_train_2[tr2]==y_train_4[tr4]:
           Training_P35.append([tr2,tr4])
        else:
           Training_N35.append([tr2,tr4])


random.shuffle(Training_N35)
Training35 = Training_P35+Training_N35[:6*len(Training_P35)]
random.shuffle(Training35)
X35=np.zeros([len(Training35),50])
X5_3=np.zeros([len(Training35),50])


y35=np.zeros([len(Training35)])
y5_3=np.zeros([len(Training35)])

yc35=np.zeros([len(Training35)])

for i in range(len(Training35)):
    in3,in5=Training35[i]
    X35[i,:]=X_train_2[in3,:]
    X5_3[i,:]=X_train_4[in5,:]

    y35[i]=y_train_2[in3]
    y5_3[i]=y_train_4[in5]
    if y_train_2[in3]==y_train_4[in5]:
        yc35[i]=1

del Training_N35,Training_P35,Training35
gc.collect()


Training_P45=[]
Training_N45=[]


for tr3 in range(len(y_train_3)):
    for tr4 in range(len(y_train_4)):

        if y_train_3[tr3]==y_train_4[tr4]:
           Training_P45.append([tr3,tr4])
        else:
           Training_N45.append([tr3,tr4])


random.shuffle(Training_N45)
Training45 = Training_P45+Training_N45[:6*len(Training_P45)]
random.shuffle(Training45)
X45=np.zeros([len(Training45),50])
X5_4=np.zeros([len(Training45),50])


y45=np.zeros([len(Training45)])
y5_4=np.zeros([len(Training45)])

yc45=np.zeros([len(Training45)])

for i in range(len(Training45)):
    in4,in5=Training45[i]
    X45[i,:]=X_train_3[in4,:]
    X5_4[i,:]=X_train_4[in5,:]

    y45[i]=y_train_3[in4]
    y5_4[i]=y_train_4[in5]
    if y_train_3[in4]==y_train_4[in5]:
        yc45[i]=1

del Training_N45,Training_P45,Training45
gc.collect()

nb_classes = len(all_list)
X12 = X12.reshape(X12.shape[0], 50, 1)
X13 = X13.reshape(X13.shape[0], 50, 1)
X14 = X14.reshape(X14.shape[0], 50, 1)
X15 = X15.reshape(X15.shape[0], 50, 1)
X23 = X23.reshape(X23.shape[0], 50, 1)
X24 = X24.reshape(X24.shape[0], 50, 1)
X25 = X25.reshape(X25.shape[0], 50, 1)
X2 = X2.reshape(X2.shape[0], 50, 1)
X3 = X3.reshape(X3.shape[0], 50, 1)
X34 = X34.reshape(X34.shape[0], 50, 1)
X35 = X35.reshape(X35.shape[0], 50, 1)
X4 = X4.reshape(X4.shape[0], 50, 1)
X5 = X5.reshape(X5.shape[0], 50, 1)
X45 = X45.reshape(X45.shape[0], 50, 1)
X3_2 = X3_2.reshape(X3_2.shape[0], 50, 1)
X4_2 = X4_2.reshape(X4_2.shape[0], 50, 1)
X5_2 = X5_2.reshape(X5_2.shape[0], 50, 1)
X4_3 = X4_3.reshape(X4_3.shape[0], 50, 1)
X5_3 = X5_3.reshape(X5_3.shape[0], 50, 1)
X5_4 = X5_4.reshape(X5_4.shape[0], 50, 1)
X_test_0 = X0_test
y_test_0 = y0_test
X_test_0 = X_test_0.reshape(X_test_0.shape[0], 50, 1)
y_test_0 = utils.to_categorical(y_test_0, nb_classes)

X_test_1 = X1_test
y_test_1 = y1_test
X_test_1 = X_test_1.reshape(X_test_1.shape[0], 50, 1)
y_test_1 = utils.to_categorical(y_test_1, nb_classes)

X_test_2 = X2_test
y_test_2 = y2_test
X_test_2 = X_test_2.reshape(X_test_2.shape[0], 50, 1)
y_test_2 = utils.to_categorical(y_test_2, nb_classes)

X_test_3 = X3_test
y_test_3 = y3_test
X_test_3 = X_test_3.reshape(X_test_3.shape[0], 50, 1)
y_test_3 = utils.to_categorical(y_test_3, nb_classes)

X_test_4 = X4_test
y_test_4 = y4_test
X_test_4 = X_test_4.reshape(X_test_4.shape[0], 50, 1)
y_test_4 = utils.to_categorical(y_test_4, nb_classes)

y12 = utils.to_categorical(y12, nb_classes)
y13 = utils.to_categorical(y13, nb_classes)
y14 = utils.to_categorical(y14, nb_classes)
y15 = utils.to_categorical(y15, nb_classes)
y23 = utils.to_categorical(y23, nb_classes)
y24 = utils.to_categorical(y24, nb_classes)
y25 = utils.to_categorical(y25, nb_classes)
y34 = utils.to_categorical(y34, nb_classes)
y35 = utils.to_categorical(y35, nb_classes)
y45 = utils.to_categorical(y45, nb_classes)
y2 = utils.to_categorical(y2, nb_classes)
y3 = utils.to_categorical(y3, nb_classes)
y4 = utils.to_categorical(y4, nb_classes)
y5 = utils.to_categorical(y5, nb_classes)
y3_2 = utils.to_categorical(y3_2, nb_classes)
y4_2 = utils.to_categorical(y4_2, nb_classes)
y5_2 = utils.to_categorical(y5_2, nb_classes)
y4_3 = utils.to_categorical(y4_3, nb_classes)
y5_3 = utils.to_categorical(y5_3, nb_classes)
y5_4 = utils.to_categorical(y5_4, nb_classes)
model1=Create_Model()
img_rows= 50
input_shape = (img_rows, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)


# number of classes for digits classification
nb_classes = len(all_list)

# Loss = (1-alpha)Classification_Loss + (alpha)CSA
alpha = .6

# Having two branches. One for source and one for target.
processed_a = model1(input_a)
processed_b = model1(input_b)


# Creating the prediction function. This corresponds to h in the paper.
out1 = Dropout(0.5)(processed_a)
out1 = Dense(64,activation='relu')(out1)
out1 = Dense(32,activation='relu')(out1)

out1 = Dense(nb_classes)(out1)
out1 = Activation('softmax', name='classification')(out1)


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='CSA')(
    [processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': contrastive_loss},
              optimizer=tf.keras.optimizers.Adam(0.00001),
              loss_weights={'classification': 1 - alpha, 'CSA': alpha})
Acc=training_the_model(model)
model.save('./your_model.h5')
model1.save('./your_model1.h5')