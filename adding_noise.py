"""TODO(ppartha): DO NOT SUBMIT without one-line documentation for Noise_added_dialogues.

TODO(ppartha): DO NOT SUBMIT without a detailed description of
Noise_added_dialogues.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from absl import app
from absl import flags

FLAGS = flags.FLAGS

import math
import google3
import os
from os import path
import json
import cPickle
import argparse
import random
from copy import deepcopy

import tensorflow as tf
from google3.pyglib import resources


fp = open('Users/MultiWoZ/MWoZ2/data.json'
)

destn = 'Users/dataset/'
change_phrase = cPickle.load(open('Users/dataset/Datasets/WoZ/Analysis/Change_Phrase.pkl','rb'))
file2topic = cPickle.load(open('Users/dataset/Datasets/WoZ/Analysis/Filename_to_Topic.pkl','rb'))
topic2files = cPickle.load(open('Users/dataset/Datasets/WoZ/Analysis/Topic_to_Filenames.pkl','rb'))
raw_data = json.load(fp)
print(len(raw_data))
next_task = cPickle.load(open('Users/dataset/Datasets/WoZ/Analysis/Next_task_distribution.pkl','rb'))

# User provides the percentage of corruption, k,-- dialogues to prepend
# Randomly pick a file and select k% utterances (rounding off to the next even number).
# Add the utterances to D['log']
# frac denotes the fraction of dataset that is corrupted
# N correspond to number files that a single file gets mixed with

def corruptDialog(
    train_list_file='file_with_fraction_of_multitask_Dialogues_and_single_domain_dialogues',
    k=10,
    frac=0.3,
    N=50):
  sng_train = tf.gfile.Open(train_list_file)
  train_files = sng_train.readlines()
  cnt = 0
  train_files = [f.strip() for f in train_files]
  up_trainfiles = deepcopy(train_files)
  for file in train_files:
    if file in file2topic:
      curr_topic = file2topic[file]
      cnt += 1
      corrupt_cnt = 0
      print(cnt, '/', len(train_files))
      for _ in range(N):
        if tuple([curr_topic]) in next_task:
          next_topic_options = next_task[tuple([curr_topic])].keys()
          next_topic = random.choice(next_topic_options)
          if next_topic != 'bus' and (curr_topic,next_topic) in change_phrase:
            corrupt_cnt += 1
            corrupt_file = random.choice(topic2files[next_topic])
            trans_phrase = random.choice(change_phrase[(curr_topic,next_topic)])

            source_file = raw_data[file]
            pure_length = len(source_file['log'])

            n_utt = int(len(source_file['log']) * k / 100.)
            if n_utt % 2 == 1:
              n_utt += 1
            source_file['log'] = source_file['log'][:n_utt] + raw_data[corrupt_file]['log']

            raw_data.update({file + '#' + corrupt_file: source_file})
            up_trainfiles += [file + '#' + corrupt_file]

  print('Saving the modified JSON for ', N)
  fp = tf.gfile.Open(
      destn + 'Trainfile_Targetted_Noise_SNG_' + str(N) + '_' + str(k) + '.txt', 'w')
  for f_name in up_trainfiles:
    fp.write(f_name + '\n')
  fp.close()
  print(len(raw_data.keys()))
  json.dump(
      raw_data,
      tf.gfile.Open(
          destn + 'Revised_data_TS_' + str(N) + '_'+ str(k) + '.json', 'w'))

def mixWithSynthetic(synth_file = '', true_file = '', MUL = 5,N=25):
  synth = tf.gfile.Open(synth_file).readlines()
  real = tf.gfile.Open(true_file).readlines()
  synth = [f.strip() for f in synth]
  real  = [f.strip() for f in real]
  train = list(set(synth).union(set(real)))
  print(len(synth),len(real),len(train))
  fp = tf.gfile.Open(
      destn + 'Synth_n_Real_Trainfile_MUL_' + str(MUL)+'_'+str(N)+'.txt', 'w')
  for file in train:
    fp.write(file+'\n')
  fp.close()

if __name__ == '__main__':
  corruptDialog(k=20, frac=1.0, N=15)
