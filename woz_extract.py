"""
   The functions defined in this file quantitatively analyzes the MultiWoZ
   dataset by getting the exact counts on multi domain and single domain
   dialogues.
   This is done by constructing a local graph from dialogue metadata, available
   in the dataset.
   This file also labels every utterance to the corresponding slot information
   obtained.
   This file also has functions that tracks information flow that happens within
   a conversation (same slot information being asked for different tasks).
   This file computes the TF and IDf scores for understanding vocabulary
   distribution.
   This file identifies the change phrase from the user that initiates task
   change. This is done for analyzing how the change phrase occurs at the
   beginning versus how it happens later on.
   This file fetches the frequencies of N-grams to identify if the model is
   overfitting to the most likely sequence or n-grams.
   This file also provides topic level tags for every dialogue.
"""

from copy import deepcopy

import math
import os
from os import path
import numpy as np
import json
import _pickle as cPickle
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.util import ngrams
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain",type=str, default="MUL")
parser.add_argument("--type",type=str, default="train")

args = parser.parse_args()
fp = open(
    '/Users/prasanna/Documents/Meta-Learning-for-Dialogue/Dataset/MULTIWOZ2/data.json')
raw_data = json.load(fp)
e_in = 0
vin = 3
Vocab = {}
Vocab['<go>'] = 0
Vocab['<eos>'] = 1
Vocab['<unk>'] = 2

# Edges are labeled and that are stored for constructing the dialog graph.

E_vocab = {}
task_count = {}
Dial_count = {}


def computeTFIDF(TF_scores, IDF_scores):
  TFIDF_scores = {}
  print(len(TF_scores))
  for j in IDF_scores:
    for i in TF_scores:
      if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
        TFIDF_scores.update({i['key']: j['IDF_score'] * i['TF_score']})
  return TFIDF_scores


def computeIDF(doc_info, freqdict_list):
  IDF_scores = []
  counter = 0
  for dict in freqdict_list:
    counter += 1
    for k in dict['freq_dict'].keys():
      count = sum([k in tempDict['freq_dict'] for tempDict in freqdict_list])
      temp = {
          'doc_id': counter,
          'IDF_score': math.log(len(doc_info) / count),
          'key': k
      }
      IDF_scores.append(temp)
  return IDF_scores


def computeTF(doc_info, freqdict_list):
  TF_scores = []
  for tempDict in freqdict_list:
    id = tempDict['doc_id']
    for k in tempDict['freq_dict']:
      temp = {
          'doc_id':
              id,
          'TF_score':
              float(tempDict['freq_dict'][k]) / doc_info[id - 1]['doc_length'],
          'key':
              k
      }
      TF_scores.append(temp)
  return TF_scores


def get_doc_info(docs):
  doc_info = []
  i = 0
  for doc in docs:
    i += 1
    count = len(word_tokenize(doc))
    temp = {'doc_id': i, 'doc_length': count}
    doc_info.append(temp)
  return doc_info


def freq_list(sents):
  i = 0
  freqDict_list = []
  for sent in sents:
    i += 1
    freq_dict = {}
    words = word_tokenize(sent)
    for word in words:
      if word not in freq_dict:
        if word == '.':
          word = '<eos>'
        freq_dict[word] = 1
      else:
        freq_dict[word] += 1
    temp = {'doc_id': i, 'freq_dict': freq_dict}
    freqDict_list.append(temp)
  return freqDict_list


def get_n_gram_stats(documents, n=4):
  """This is to analyze the n-gram distribution in the dataset.
  """
  N_grams = {}
  for sent in documents:
    #print(sent)
    sent = sent.lower()
    sent = re.sub(r'[^a-zA-Z0-9\s]', ' ', sent)
    tokens = [token for token in sent.split(' ') if token != '']
    ngrams_all = list(ngrams(tokens, n))
    #print(sent,ngrams_all)
    for ngram in ngrams_all:
      if ngram in N_grams:
        N_grams[ngram] += 1
      else:
        N_grams[ngram] = 1
  return N_grams


def getdocs(D, type_='utterance'):  #Pass the data dictionary
  docs = []
  for i in D.keys():
    str_doc = ''
    for j in D[i]['log'].keys():
      if 'user utterance' in D[i]['log'][j]:
        str_doc += ' '.join(D[i]['log'][j]['user utterance'])
      elif 'machine utterance' in D[i]['log'][j]:
        str_doc += ' '.join(D[i]['log'][j]['machine utterance'])
    docs += [str_doc]
  return docs


def extract_utterance_info(level, key, root, D={}, D_local={}):
  """This recursion function on the meta data, parses the meta data corresponding to an occurrence and extracts the information in (relation, node).

  global variable task_count stores the utterance level statistics of
  dialogues.
  """
  global Vocab, E_vocab, vin, e_in, task_count
  if key != '' and len(key.split('#')) < 2:
    if key.split('#')[0] not in task_count:
      task_count[key.split('#')[0]] = 1
    else:
      task_count[key.split('#')[0]] += 1
  if 'dict' not in str(type(level)):
    return True
  else:
    keys = list(level.keys())
    for k in keys:
      key = root + '#' + k
      if 'dict' not in str(type(level[k])):
        if 'list' not in str(type(level[k])) and level[k] != '' and len(
            level[k]) != 0 and level[k] != 'not mentioned':
          if key not in D:
            D_local.update({key: level[k]})
            D.update({key: level[k]})
            if key.strip() not in E_vocab:
              E_vocab.update({key.strip(): e_in})
              e_in += 1
      extract_utterance_info(level[k], k, key, D, D_local)
      key = root
  return D_local, D


def extract_info_from_dialogues(file_):
  """This method iterates over every utterance in every dialogue to construct the graph corresponding to every dialogue.
  """
  global Vocab, E_vocab, vin, e_in
  raw_data[file_]['goal']['message'] = ' '.join(
      raw_data[file_]['goal']['message']).replace('</span>', '')
  raw_data[file_]['goal']['message'] = raw_data[file_]['goal'][
      'message'].replace('<span class=\'emphasis\'>', '')
  message = word_tokenize(raw_data[file_]['goal']['message'])
  message_ = []
  Dialog = {}
  for line in message:
    line_ = line.replace('.', ' <eos> ')
    l = []
    for word in word_tokenize(line):
      if word.lower().strip() not in Vocab:
        Vocab.update({word.lower().strip(): vin})
        vin += 1
      l += [word.lower().strip()]
    message_ += l
  D_global = {}
  tc_local = {}
  max_ut_len = 0
  for i in range(len(raw_data[file_]['log']) - 1):
    text = raw_data[file_]['log'][i]['text']
    text_ = []
    for token in word_tokenize(text):
      if token.lower().strip() not in Vocab:
        Vocab.update({token.lower().strip(): vin})
        vin += 1
      text_ += [token.lower().strip()]
    if len(text_) > max_ut_len:
      max_ut_len = len(text_)
    D_local = {}
    if i < len(raw_data[file_]['log']):
      D_local, D_global = extract_utterance_info(raw_data[file_]['log'][i + 1]['metadata'], '',
                                '', deepcopy(D_global), D_local)
      if len(D_local) > 0:
        for k_ in D_local.keys():
          task = k_.split('#')[1]
          slot = k_.split('#')[-1]
          if slot not in tc_local:
            tc_local[slot] = [{'utterance': i, 'task': task}]
          else:
            tc_local[slot] += [{'utterance': i, 'task': task}]
    else:
      D_local = {}
    if i == 0:
      Dialog.update({
          'log': {
              i: {
                  'user utterance': text_,
                  'local_graph': D_local,
                  'global': deepcopy(D_global)
              }
          }
      })
    else:
      if i % 2:
        Dialog['log'].update({
            i: {
                'machine utterance': text_,
                'local_graph': D_local,
                'global': deepcopy(D_global)
            }
        })
      else:
        Dialog['log'].update({
            i: {
                'user utterance': text_,
                'local_graph': D_local,
                'global': deepcopy(D_global)
            }
        })
    #print(tc_local)
    Dialog.update({
        'template': {
            'message': message_,
            'graph': D_global
        },
        'stats': deepcopy(tc_local)
    })

  return Dialog, len(message_), len(D_global), max_ut_len


def getInfoOverlap(D):
  """This method extgracts information about overlap in slots across the tasks in a multitask dialogue.
  """
  overlap = {}
  for fname, dial in D.items():
    stats = dial['stats']
    for k, occurrences in stats.items():
      if len(occurrences) > 1:
        tasks = []
        utterances = []
        for occ in occurrences:
          tasks += [occ['task']]
          utterances += [occ['utterance']]
        if fname not in overlap:
          overlap[fname] = [{
              'slot': k,
              'tasks': tasks,
              'utterances': utterances
          }]
        else:
          overlap[fname] += [{
              'slot': k,
              'tasks': tasks,
              'utterances': utterances
          }]

  return overlap


def getChangePhrase(D):
  """This method extracts the utterance by the user that caused a topic change.

  Also it records the current topic and the changed topic.
  """
  changephrases = {}
  for fname, dial in D.items():
    prevtopic = None
    prev_utterance = None
    for utt_ in range(len(dial['log'])):
      for k in dial['log'][utt_]['local_graph'].keys():
        if prevtopic is None or k.split('#')[1] != prevtopic:
          if 'user utterance' in dial['log'][utt_]:
            curr_utterance = deepcopy(dial['log'][utt_]['user utterance'])
          else:
            curr_utterance = deepcopy(dial['log'][utt_]['machine utterance'])
          if fname not in changephrases:
            changephrases[fname] = {
                utt_: {
                    'from': deepcopy(prevtopic),
                    'to': deepcopy(k.split('#')[1]),
                    'utt_prev': deepcopy(prev_utterance),
                    'utt_curr': deepcopy(curr_utterance)
                }
            }
          else:
            changephrases[fname].update({
                utt_: {
                    'from': deepcopy(prevtopic),
                    'to': deepcopy(k.split('#')[1]),
                    'utt_prev': deepcopy(prev_utterance),
                    'utt_curr': deepcopy(curr_utterance)
                }
            })
          #print(changephrases[fname])
      if 'user utterance' in dial['log'][utt_]:
        prevtopic = deepcopy(k.split('#')[1])
      if 'user utterance' in dial['log'][utt_]:
        prev_utterance = deepcopy(dial['log'][utt_]['user utterance'])
      else:
        prev_utterance = deepcopy(dial['log'][utt_]['machine utterance'])
  return changephrases


def get_dial_stats(D):
  """This function extracts information about which dialogue files discusses a particular topic. The topics are high level.

  'restaurant', 'taxi'.
  """
  dial_stats_global = {}
  for fname, dial in D.items():
    dial_stats_local = {}
    for k in dial['log'][len(dial['log']) - 1]['global'].keys():
      if k.split('#')[1] not in dial_stats_local:
        dial_stats_local[k.split('#')[1]] = set([fname])
    for k in dial_stats_local.keys():
      if k in dial_stats_global:
        dial_stats_global[k] = dial_stats_global[k].union(dial_stats_local[k])
      else:
        dial_stats_global[k] = dial_stats_local[k]
  return dial_stats_global


def dial_combinations(dial_stats):
  dial_comb = {}  # dialogue index level information for compositionality
  comb_stats = {}  # overall count for k-task dialogues
  comb_numbers = {}  # exact numbers for combinations present
  for k, v in dial_stats.items():
    topic = k
    for idx_ in list(v):
      if idx_ in dial_comb:
        dial_comb[idx_] = dial_comb[idx_].union([topic])
      else:
        dial_comb[idx_] = set([topic])
  for k, v in dial_comb.items():
    if frozenset(v) in comb_numbers:
      comb_numbers[frozenset(v)] += 1
    else:
      comb_numbers[frozenset(v)] = 1
    if len(v) in comb_stats:
      comb_stats[len(v)] += 1
    else:
      comb_stats[len(v)] = 1

  return dial_comb, comb_numbers, comb_stats


def main(FLAGS):
  print(FLAGS.domain + ' domain dialogues on MultiWoZ dataset')
  Files = list(raw_data.keys())
  Dataset = {}
  max_mess_len = 0
  max_graph_size = 0
  max_utt_len = 0
  f_ = open(
      '/Users/prasanna/Documents/Meta-Learning-for-Dialogue/Dataset/MULTIWOZ2/testListFile.json'
  )
  f_test = f_.readlines()
  f_ = open(
      '/Users/prasanna/Documents/Meta-Learning-for-Dialogue/Dataset/MULTIWOZ2/valListFile.json'
  )
  f_valid = f_.readlines()
  for i in range(len(f_valid)):
    f_valid[i] = f_valid[i].strip()
    f_test[i] = f_test[i].strip()

  print('Files are being opened and read ...')
  if FLAGS.type == 'train':
    to_eval = 'f not in f_valid and f not in f_test'
  else:
    to_eval = 'f in f_valid'
  for f in Files:
    if eval(to_eval) and FLAGS.domain in f:
      i_D, m_, d_len, u_len = extract_info_from_dialogues(f)
      if max_mess_len < m_:
        max_mess_len = m_
      if max_graph_size < d_len:
        max_graph_size = d_len
      if max_utt_len < u_len:
        max_utt_len = u_len
      Dataset.update({f: i_D})
  print('m_mess_len:', max_mess_len)
  print('m_graph_size:', max_graph_size)
  print('m_ut_len:', max_utt_len)
  print('Vocab size:', len(Vocab))

  print('Document preparation for TF-IDF computation ...')
  docs = getdocs(Dataset)

  print('Frequency List preparation for TF-IDF computation ...')
  freq_list_ = freq_list(docs)
  N_grams_freq = get_n_gram_stats(docs, n=4)
  print(
      'Preparing Document information (word in document) for TF-IDF computation ...'
  )
  doc_info = get_doc_info(docs)

  print('task Level label counts: ', task_count)

  print('Dialogue statistics being computed...')
  dial_stats = get_dial_stats(Dataset)
  overlapinfo = getInfoOverlap(Dataset)
  changephrase = getChangePhrase(Dataset)
  dial_comb, comb_numbers, comb_stats = dial_combinations(dial_stats)

  print('Done computing all the stats ... Now saving ..')

  destn_folder = './'
  cPickle.dump(
      overlapinfo,
      open(
          destn_folder + 'Information_overlap_multitask_' + FLAGS.type + '.pkl', 'wb'))
  print('Saved Overlap Info ..')

  cPickle.dump(
      changephrase,
      open(
          destn_folder + 'Topic_Change_Phrase_' + FLAGS.type + '_' + FLAGS.domain +
          '.pkl', 'wb'))
  print('Saved Change phrase..')

  cPickle.dump(
      N_grams_freq,
      open(
          destn_folder + 'N_grams_' + FLAGS.type + '_' + FLAGS.domain + '.pkl', 'wb'))
  print('Saved N-grams frequency information')

  cPickle.dump(
      dial_comb,
      open(
          destn_folder + 'Dialogue_combinations_' + FLAGS.type + '_' + FLAGS.domain +
          '.pkl', 'wb'))
  print('Saved Combination counts saved ..')

  cPickle.dump(
      task_count,
      open(
          destn_folder + 'Utterance_level_classification_' + FLAGS.type + '_' +
          FLAGS.domain + '.pkl', 'wb'))
  print('Utterance level labels saved ...')

  cPickle.dump(
      comb_numbers,
      open(
          destn_folder + 'Exact_combinations_' + FLAGS.type + '_' + FLAGS.domain +
          '.pkl', 'wb'))
  print('Exact numbers for combinations saved ..')

  cPickle.dump(
      comb_stats,
      open(
          destn_folder + 'Combinations_stats_' + FLAGS.type + '_' + FLAGS.domain +
          '.pkl', 'wb'))
  print('Combination stats saved ...')

  print('Final.. Saving the dataset..')
  cPickle.dump(
      Dataset,
      open(
          destn_folder+'Datasets/WoZ/Dataset_'
          + FLAGS.domain + '_WoZ_train.pkl', 'wb'))
  cPickle.dump(
      E_vocab,
      open(
          destn_folder+'Datasets/WoZ/Edges_'
          + FLAGS.domain + '_Woz_train.pkl', 'wb'))
  cPickle.dump(
      Vocab,
      open(
          destn_folder+'Datasets/WoZ/Vocab_'
          + FLAGS.domain + '_Woz_train.pkl', 'wb'))
  print('Done!')


if __name__ == '__main__':
  main(args)
