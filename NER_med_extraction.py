"""
Created Date:       11/1/2024
Last update:        13/5/2024
Program:            Medication-extraction-revised
@author:            Wachiranun Sirikul, MD., M.Sc.
                    Natthanaphop Isaradech, MD.
Description:        NER tasks using pre-trained models

Part 1: Package installaion
Part 2: Library and function import
Part 3: Import the models from pickle
Part 4: Check GPU is ready to use
Part 5: Weigth and bias tracking set-up
Part 6: Entity labels
Part 7: Import datasets
Part 8: Modeling
- Model 1: emilyalsentzer/Bio_ClinicalBERT
- Model 2: Microsoft biomedNLP
- Model 3: medicalai/ClinicalBERT
"""

"""Part 1: Package installaion (if needed)"""
# !pip install simpletransformers transformers torch
# !pip install wandb
# !pip install eval4ner
# # Spacy transformer model download
# !python -m spacy download en_core_web_trf
# !pip install svgutils
# !pip install cairosvg
# !pip install tksvg

"""Part 2: Library and function import"""
# Tacking activities
import logging
import wandb
import json
import codecs
# NER model development and validation
from simpletransformers.ner import NERModel, NERArgs
import torch

# General packages and data pre-processing
import eval4ner.muc as muc
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import svgutils.compose as sc
import pprint

# Performance evaluation
from sklearn.metrics import precision_score , recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from copy import deepcopy
from collections import defaultdict

# Spacy packages for NER visualization
import spacy
from spacy.tokens import Doc # Making Spacy Doc from our predictions
from spacy import displacy # Making NER visualization

# Function for check unique in list
from functools import reduce
def unique(list1):
    # Print directly by using * symbol
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)

"""Part 3:  Import the models from pickle"""
# All models
"-------------------------Getting obj back-------------------------------"
with open('allmodels.pkl', 'rb') as f:
    model, model2, model3 = pickle.load(f)

"""Part 4:  Check GPU is ready to use"""
torch.cuda.is_available()

"""Part 5:  Weigth and bias tracking set-up"""
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Medication-extraction-revised",
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

"""Part 6:  Entity labels"""
list = ['MED',
'PRO',
'UNI',
'SUB',
'TIM',
'DEC',
'ROU',
'MEAS',
'RTIM']
#for k, v in keys.items():
 #   list.append(v)
labels = []
for i in list:
    label = "B-" + i
    label2 = "I-" + i
    labels.append(label)
    labels.append(label2)
labels.append("O")

"""Part 7: Import datasets and test-set preparation"""
# Replace with your specific data paths and labels list (using conll format)
training_set = "TRAIN.conll" 
validation_set = "VALIDATION.conll"
test_set = "TEST.conll"
labels = labels

# import test set
file_path_test_set = test_set

# Initialize empty lists for sentences and labels
sentences = []
labels_ground_truth = []
sentence = []
sentence_labels = []
file_lines = []
muc_ground_truth = []
muc_ground_truth_sentence = []

# Read the CoNLL dataset from file
with open(file_path_test_set, 'r',encoding="utf8") as file:
    for line in file:
        # Split each line into token and label using whitespace as a separator
        line = line.strip()

        if line:  # skip empty lines
            file_lines.append(line)
            parts = line.strip().split()
            token = parts[0].strip()
            labelx = parts[1].strip()
            muc_ground_truth_sentence.append((labelx, token))

            sentence.append(token)
            sentence_labels.append(labelx)

        else:
            # Add completed sentence and its labels to the lists
            if sentence:
                sentences.append(' '.join(sentence))              
                labels_ground_truth.append(sentence_labels)
                muc_ground_truth.append(muc_ground_truth_sentence)
                            
            # Reset current sentence and its labels
            sentence = []
            sentence_labels = []
            muc_ground_truth_sentence = []
   
"""Part 8: Modeling"""
"""///////////// Model 1: emilyalsentzer/Bio_ClinicalBERT //////////////////
"""
# logging information 

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model arguments
model_args = NERArgs(num_train_epochs = 10, overwrite_output_dir = True)
model_args.labels_list = labels
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 2
model_args.focal_loss = True
model_args.class_weight = "balanced"
# model output path
model_args.output_dir = "Bio_ClinicalBERT/"
model_args.wandb_project = "Medication-extraction-revised"

model = NERModel("bert",
    "emilyalsentzer/Bio_ClinicalBERT",
    args=model_args,
    use_cuda=True,
    labels=labels)

model.train_model(training_set, eval_data=validation_set)

result_eva1, model_outputs_eva1, preds_list_eva1 = model.eval_model(validation_set)

wandb.finish()

# Evaluate the model
result_model_test1, model_outputs_test1, preds_list_test1 = model.eval_model(test_set)

""" Make an annotation with test sentences """
# evaluation and predictions with model *validation*
predictions, raw_outputs = model.predict(sentences)

# evaluation and predictions with model *test set*
predictions, raw_outputs = model.predict(sentences)


predictions_list = []
predictions_dict = []
for i in predictions:
    for j in i:
        for k, v in j.items():
            x  = (v,k)
            predictions_dict.append(x)
    predictions_list.append(predictions_dict)
    predictions_dict = []

result_overall = muc.evaluate_all(predictions_list, muc_ground_truth * 1, sentences, verbose=True)
pprint.pprint(result_overall)

# Change predictions taging for NER visualization 
y_tag = pd.Series(y_pred, index=None) 
y_tag.replace({'B-SUB': 'B-BSUB',
                'B-DEC': 'B-BDEC',
                'B-ROU': 'B-BROU', 
                'B-TIM': 'B-BTIM', 
                'B-UNI': 'B-BUNI',
                'I-TIM': 'I-ITIM',
                'I-SUB': 'I-ISUB'},inplace = True)
y_tag = y_tag.tolist()
print(y_tag)

""" Prepare data for NER visualization and performance evaluation (Reference set)"""
# Re-shape references --> 1-d list of tuple
true_token = []
y_true = []
for j in muc_ground_truth:
    true = j[0]
    token = j[1]
    true_token.append(token)
    y_true.append(true)

# Change reference taging for NER visualization 
y_true_tag = pd.Series(y_true)
y_true_tag.replace({'B-SUB': 'B-BSUB',
                'B-DEC': 'B-BDEC',
                'B-ROU': 'B-BROU', 
                'B-TIM': 'B-BTIM', 
                'B-UNI': 'B-BUNI',
                'I-ROU': 'I-IROU',
                'I-TIM': 'I-ITIM',
                'I-SUB': 'I-ISUB',
                'B-MEAS': 'B-BMEAS',
                'B-RTIM': 'B-RTIM'}, inplace = True)
y_true_tag = y_true_tag.tolist()
print(y_true_tag)


""" Create dataframe for confusion matrix """
# Create dataframe from lists of reference and predictions
df_true = pd.DataFrame(data = {'token' :true_token,
                               'true':y_true})

df_pred = pd.DataFrame(data = {'token' :pred_token,
                               'pred':y_pred})

y_concat =  pd.merge(df_true, df_pred, how='right', on= 'token')
y_concat_dropdu = y_concat.drop_duplicates('token',keep='last')


y_concat_no_o = y_concat.replace({'O':np.nan})
y_concat_no_o.dropna(inplace = True)
y_concat_dropdu_no_o = y_concat_no_o.drop_duplicates('token',keep='last')

### y_concat_dropdu["pred"].fillna(y_concat_dropdu["true"], inplace=True)

# list of reference for making a confusion matrix
y_true_cf = y_concat_dropdu['true']

y_true_no_bio = y_concat_dropdu_no_o['true']
y_true_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)
# list of predictions for making a confusion matrix
y_pred_cf = y_concat_dropdu['pred']

y_pred_no_bio = y_concat_dropdu_no_o['pred']
y_pred_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)

""" Create confusion matrix of test-set performance """
# Create confusion matrix of test-set performance
cm = confusion_matrix(y_true_cf, y_pred_cf, labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])

cm_no_bio = confusion_matrix(y_true_no_bio, y_pred_no_bio, labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])

# Create confusion matrix plot of test-set performance
### Confusion matrix model 1 (all classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(7.5) 
fig.suptitle('Confusion matrix plot of BioClinicalBERT')

### Confusion matrix model 1 (No B-I-O)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_no_bio, display_labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(6) 
fig.suptitle('Confusion matrix plot of BioClinicalBERT')

""" Test-set performance by class labels """
# Test performance
## (all classes) ##
test_byclass = precision_recall_fscore_support(y_true_cf, y_pred_cf,
                                               labels= ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
# Create dataframe of test performance
df_precision_recall_fscore_support = pd.DataFrame(test_byclass,columns = ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS','B-RTIM' ], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df_precision_recall_fscore_support)

test_macro = precision_recall_fscore_support(y_true_cf, y_pred_cf, average = 'macro')
test_micro = precision_recall_fscore_support(y_true_cf, y_pred_cf, average = 'micro')
test_weighted = precision_recall_fscore_support(y_true_cf, y_pred_cf, average = 'weighted')


## (No B-I-O) ##
test_byclass_no_bio = precision_recall_fscore_support(y_true_no_bio, y_pred_no_bio,
                                                      labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'] )
# Create dataframe of test performance
df_precision_recall_fscore_support_no_bio = pd.DataFrame(test_byclass_no_bio,columns = ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df_precision_recall_fscore_support_no_bio)

test_macro_no_bio = precision_recall_fscore_support(y_true_no_bio, y_pred_no_bio, average = 'macro')
test_micro_no_bio = precision_recall_fscore_support(y_true_no_bio, y_pred_no_bio, average = 'micro')
test_weighted_no_bio = precision_recall_fscore_support(y_true_no_bio, y_pred_no_bio, average = 'weighted')

# Copy table of performance to clipboard
df_precision_recall_fscore_support.to_clipboard(excel=True,sep='\t')
df_precision_recall_fscore_support_no_bio.to_clipboard(excel=True,sep='\t')

"""
Construction Doc spacy and NER visualization for reference annotations
"""
# load default transformer model from spacy
nlp = spacy.load("en_core_web_trf")

# Check unique reference tagging
print(unique(y_true))

# Doc of reference annotations
doc_true = Doc(nlp.vocab, words=true_token, ents=y_true)

#checking entity in doc
for entity in doc_true.ents:
    print(entity.text, entity.label_)

# Set-up color palette
colors = {"SUB": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
         "DEC":  "#BED7DC",
         "ROU": "#EEA5A6",
         "TIM": "linear-gradient(90deg, yellow)",
         "UNI": "#E2F4C5"}
options = {"ents": ["SUB", "DEC", "ROU", "TIM","UNI", "MEAS",'RTIM'], "colors": colors }

# Render NER visualization for reference annotations
html = displacy.render(doc_true, style="ent", options = options, page = True)
output_path = Path("model1_reference.html")
output_path.open("w", encoding="utf-8").write(html)

"""
Construction Doc spacy and NER visualization for reference annotations
"""
# load default transformer model from spacy
nlp = spacy.load("en_core_web_trf")

# Check unique reference tagging
print(unique(y_true))

# Doc of reference annotations
doc_pred = Doc(nlp.vocab, words=pred_token, ents=y_pred)

#checking entity in doc
for entity in doc_pred.ents:
    print(entity.text, entity.label_)

# Set-up color palette
colors = {"SUB": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
         "DEC":  "#BED7DC",
         "ROU": "#EEA5A6",
         "TIM": "linear-gradient(90deg, yellow)",
         "UNI": "#E2F4C5"}
options = {"ents": ["SUB", "DEC", "ROU", "TIM","UNI", "MEAS",'RTIM'], "colors": colors }

# Render NER visualization for reference annotations
html = displacy.render(doc_pred, style="ent", options = options, page = True)
output_path = Path("model1_pred.html")
output_path.open("w", encoding="utf-8").write(html)

# Save model (Bio_ClinicalBERT)
"----------------------------Save Varaibles------------------------------"
with open('C:/Users/AMG/OneDrive - Chiang Mai University/Documents/FEN/KG_NER/REVISED_DATASET2/Bio_ClinicalBERT.pkl', 'wb') as f:
    pickle.dump(model, f)
    
"""/////////////////////////////////////////////////////////////////////////////////"""
"""///////////// Model 2: BiomedNLP //////////////////"""

# logging information 

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model arguments
model_args = NERArgs(num_train_epochs = 10, overwrite_output_dir = True)
model_args.labels_list = labels
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 2
model_args.focal_loss = True
model_args.class_weight = "balanced"
# Model output path
model_args.output_dir = "BiomedNLP/"
model_args.wandb_project = "medication-extraction"

model2 = NERModel("bert",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    args=model_args,
    use_cuda=True,
    labels=labels)

model2.train_model(training_set, eval_data=validation_set)
wandb.finish()

# Evaluate the model
result_model_test2, model_outputs_test2, preds_list_test2 = model2.eval_model(test_set)

""" Make an annotation with test sentences """
# evaluation and predictions with model 
predictions2, raw_outputs2 = model2.predict(sentences)

predictions_list2 = []
predictions_dict2 = []
for i in predictions2:
    for j in i:
        for k, v in j.items():
            x  = (v,k)
            predictions_dict2.append(x)
    predictions_list2.append(predictions_dict2)
    predictions_dict2 = []

result_overall2 = muc.evaluate_all(predictions_list2, muc_ground_truth * 1, sentences, verbose=True)
pprint.pprint(result_overall2)

"""Prepare data for NER visualization and performance evaluation (Test predictions)"""
# Re-shape predictions --> 1-d list
pred_token2 = []
y_pred2 = []
for i in predictions2:
        for j in i:
            key, value = next(iter(j.items()))
            pred_token2.append(key)
            y_pred2.append(value)

# Change predictions taging for NER visualization 
y_tag2 = pd.Series(y_pred2, index=None) 
y_tag2.replace({'B-SUB': 'B-BSUB',
                'B-DEC': 'B-BDEC',
                'B-ROU': 'B-BROU', 
                'B-TIM': 'B-BTIM', 
                'B-UNI': 'B-BUNI',
                'I-TIM': 'I-ITIM',
                'I-SUB': 'I-ISUB'},inplace = True)
y_tag2 = y_tag2.tolist()
print(y_tag2)


""" Create dataframe for confusion matrix """
# Create dataframe from lists of reference and predictions
df_true = pd.DataFrame(data = {'token' :true_token,
                               'true':y_true})

df_pred2 = pd.DataFrame(data = {'token' :pred_token2,
                               'pred':y_pred2})

y_concat2 =  pd.merge(df_true, df_pred2, how='right', on= 'token')
y_concat2_dropdu = y_concat2.drop_duplicates('token',keep='last')


y_concat2_no_o = y_concat2.replace({'O':np.nan})
y_concat2_no_o.dropna(inplace = True)
y_concat2_dropdu_no_o = y_concat2_no_o.drop_duplicates('token',keep='last')


# list of reference for making a confusion matrix
y_true2_cf = y_concat2_dropdu['true']

y_true2_no_bio = y_concat2_dropdu_no_o['true']
y_true2_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)
# list of predictions for making a confusion matrix
y_pred2_cf = y_concat2_dropdu['pred']

y_pred2_no_bio = y_concat2_dropdu_no_o['pred']
y_pred2_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)

""" Create confusion matrix of test-set performance """
# Create confusion matrix of test-set performance
cm = confusion_matrix(y_true2_cf, y_pred2_cf, labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])

cm_no_bio = confusion_matrix(y_true2_no_bio, y_pred2_no_bio, labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])

# Create confusion matrix plot of test-set performance
### Confusion matrix model 2 (all classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(7.5) 
fig.suptitle('Confusion matrix plot of BiomedNLP')

### Confusion matrix model 2 (No B-I-O)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_no_bio, display_labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(6) 
fig.suptitle('Confusion matrix plot of BiomedNLP')

"""Test-set performance by class labels """
# Test performance
## (all classes) ##
test2_byclass = precision_recall_fscore_support(y_true2_cf, y_pred2_cf,
                                               labels= ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
# Create dataframe of test performance
df2_precision_recall_fscore_support = pd.DataFrame(test2_byclass,columns = ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS','B-RTIM' ], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df2_precision_recall_fscore_support)

test2_macro = precision_recall_fscore_support(y_true2_cf, y_pred2_cf, average = 'macro')
test2_micro = precision_recall_fscore_support(y_true2_cf, y_pred2_cf, average = 'micro')
test2_weighted = precision_recall_fscore_support(y_true2_cf, y_pred2_cf, average = 'weighted')


## (No B-I-O) ##
test2_byclass_no_bio = precision_recall_fscore_support(y_true2_no_bio, y_pred2_no_bio,
                                                      labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'] )
# Create dataframe of test performance
df2_precision_recall_fscore_support_no_bio = pd.DataFrame(test2_byclass_no_bio,columns = ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df2_precision_recall_fscore_support_no_bio)

test2_macro_no_bio = precision_recall_fscore_support(y_true2_no_bio, y_pred2_no_bio, average = 'macro')
test2_micro_no_bio = precision_recall_fscore_support(y_true2_no_bio, y_pred2_no_bio, average = 'micro')
test2_weighted_no_bio = precision_recall_fscore_support(y_true2_no_bio, y_pred2_no_bio, average = 'weighted')

# Copy table of performance to clipboard
df2_precision_recall_fscore_support.to_clipboard(excel=True,sep='\t')
df2_precision_recall_fscore_support_no_bio.to_clipboard(excel=True,sep='\t')


"""
Construction Doc spacy and NER visualization for reference annotations
"""
# load default transformer model from spacy
nlp = spacy.load("en_core_web_trf")

# Check unique reference tagging
print(unique(y_pred2))

# Doc of reference annotations
doc_pred2 = Doc(nlp.vocab, words=pred_token2, ents=y_pred2)

#checking entity in doc
for entity in doc_pred2.ents:
    print(entity.text, entity.label_)

# Set-up color palette
colors = {"SUB": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
         "DEC":  "#BED7DC",
         "ROU": "#EEA5A6",
         "TIM": "linear-gradient(90deg, yellow)",
         "UNI": "#E2F4C5"}
options = {"ents": ["SUB", "DEC", "ROU", "TIM","UNI", "MEAS",'RTIM'], "colors": colors }

# Render NER visualization for reference annotations
html = displacy.render(doc_pred2, style="ent", options = options, page = True)
output_path = Path("model2_pred.html")
output_path.open("w", encoding="utf-8").write(html)

    
"""/////////////////////////////////////////////////////////////////////////////////"""
"""///////////// Model 3: medicalai/ClinicalBERT //////////////////"""

# logging information 

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model arguments
model_args = NERArgs(num_train_epochs = 10, overwrite_output_dir = True)
model_args.labels_list = labels
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_patience = 2
model_args.focal_loss = True
model_args.class_weight = "balanced"
# Model output path
model_args.output_dir = "ClinicalBERT/"
model_args.wandb_project = "medication-extraction"

model3 = NERModel("bert",
    "medicalai/ClinicalBERT",
    args=model_args,
    use_cuda=True,
    labels=labels)

model3.train_model(training_set, eval_data=validation_set)
wandb.finish()

# Evaluate the model
result_model_test3, model_outputs_test3, preds_list_test3 = model3.eval_model(test_set)

""" Make an annotation with test sentences """
# evaluation and predictions with model 
predictions3, raw_outputs3 = model3.predict(sentences)

predictions_list3 = []
predictions_dict3 = []
for i in predictions3:
    for j in i:
        for k, v in j.items():
            x  = (v,k)
            predictions_dict3.append(x)
    predictions_list3.append(predictions_dict3)
    predictions_dict3 = []

result_overall3 = muc.evaluate_all(predictions_list3, muc_ground_truth * 1, sentences, verbose=True)
pprint.pprint(result_overall3)

"""Prepare data for NER visualization and performance evaluation (Test predictions)"""
# Re-shape predictions --> 1-d list
pred_token3 = []
y_pred3 = []
for i in predictions3:
        for j in i:
            key, value = next(iter(j.items()))
            pred_token3.append(key)
            y_pred3.append(value)

# Change predictions taging for NER visualization 
y_tag3 = pd.Series(y_pred3, index=None) 
y_tag3.replace({'B-SUB': 'B-BSUB',
                'B-DEC': 'B-BDEC',
                'B-ROU': 'B-BROU', 
                'B-TIM': 'B-BTIM', 
                'B-UNI': 'B-BUNI',
                'I-TIM': 'I-ITIM',
                'I-SUB': 'I-ISUB'},inplace = True)
y_tag3 = y_tag3.tolist()
print(y_tag3)


""" Create dataframe for confusion matrix """
# Create dataframe from lists of reference and predictions
df_true = pd.DataFrame(data = {'token' :true_token,
                               'true':y_true})

df_pred3 = pd.DataFrame(data = {'token' :pred_token3,
                               'pred':y_pred3})

y_concat3 =  pd.merge(df_true, df_pred3, how='right', on= 'token')
y_concat3_dropdu = y_concat3.drop_duplicates('token',keep='last')


y_concat3_no_o = y_concat3.replace({'O':np.nan})
y_concat3_no_o.dropna(inplace = True)
y_concat3_dropdu_no_o = y_concat3_no_o.drop_duplicates('token',keep='last')


# list of reference for making a confusion matrix
y_true3_cf = y_concat3_dropdu['true']

y_true3_no_bio = y_concat3_dropdu_no_o['true']
y_true3_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)
# list of predictions for making a confusion matrix
y_pred3_cf = y_concat3_dropdu['pred']

y_pred3_no_bio = y_concat3_dropdu_no_o['pred']
y_pred3_no_bio.replace({'B-SUB': 'SUB',
                    'B-DEC': 'DEC',
                    'B-ROU': 'ROU', 
                    'B-TIM': 'TIM', 
                    'B-UNI': 'UNI',
                    'I-ROU': 'ROU',
                    'I-TIM': 'TIM',
                    'I-SUB': 'SUB',
                    'B-MEAS': 'MEAS',
                    'B-RTIM': 'RTIM'}, inplace = True)

""" Create confusion matrix of test-set performance """
# Create confusion matrix of test-set performance
cm = confusion_matrix(y_true3_cf, y_pred3_cf, labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])

cm_no_bio = confusion_matrix(y_true3_no_bio, y_pred3_no_bio, labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])

# Create confusion matrix plot of test-set performance
### Confusion matrix model 3 (all classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['O','B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(7.5) 
fig.suptitle('Confusion matrix plot of ClinicalBERT')

### Confusion matrix model 3 (No B-I-O)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_no_bio, display_labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'])
disp.plot()
fig = disp.figure_
fig.set_figwidth(12)
fig.set_figheight(6) 
fig.suptitle('Confusion matrix plot of ClinicalBERT')

"""Test-set performance by class labels """
# Test performance
## (all classes) ##
test3_byclass = precision_recall_fscore_support(y_true3_cf, y_pred3_cf,
                                               labels= ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS', 'B-RTIM'])
# Create dataframe of test performance
df3_precision_recall_fscore_support = pd.DataFrame(test3_byclass,columns = ['O', 'B-SUB', 'B-DEC', 'B-ROU', 'B-TIM', 'B-UNI', 'I-ROU', 'I-TIM', 'I-SUB', 'B-MEAS','B-RTIM' ], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df3_precision_recall_fscore_support)

test3_macro = precision_recall_fscore_support(y_true3_cf, y_pred3_cf, average = 'macro')
test3_micro = precision_recall_fscore_support(y_true3_cf, y_pred3_cf, average = 'micro')
test3_weighted = precision_recall_fscore_support(y_true3_cf, y_pred3_cf, average = 'weighted')


## (No B-I-O) ##
test3_byclass_no_bio = precision_recall_fscore_support(y_true3_no_bio, y_pred3_no_bio,
                                                      labels= ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'] )
# Create dataframe of test performance
df3_precision_recall_fscore_support_no_bio = pd.DataFrame(test3_byclass_no_bio,columns = ['SUB', 'DEC', 'ROU', 'TIM', 'UNI', 'ROU', 'MEAS', 'RTIM'], index = ['Precision', 'Recall' , 'F1-score' ,'Support'])
print(df3_precision_recall_fscore_support_no_bio)

test3_macro_no_bio = precision_recall_fscore_support(y_true3_no_bio, y_pred3_no_bio, average = 'macro')
test3_micro_no_bio = precision_recall_fscore_support(y_true3_no_bio, y_pred3_no_bio, average = 'micro')
test3_weighted_no_bio = precision_recall_fscore_support(y_true3_no_bio, y_pred3_no_bio, average = 'weighted')

# Copy table of performance to clipboard
df3_precision_recall_fscore_support.to_clipboard(excel=True,sep='\t')
df3_precision_recall_fscore_support_no_bio.to_clipboard(excel=True,sep='\t')


"""
Construction Doc spacy and NER visualization for reference annotations
"""
# load default transformer model from spacy
nlp = spacy.load("en_core_web_trf")

# Check unique reference tagging
print(unique(y_true))

# Doc of reference annotations
doc_pred3 = Doc(nlp.vocab, words=pred_token3, ents=y_pred3)

#checking entity in doc
for entity in doc_pred3.ents:
    print(entity.text, entity.label_)

# Set-up color palette
colors = {"SUB": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
         "DEC":  "#BED7DC",
         "ROU": "#EEA5A6",
         "TIM": "linear-gradient(90deg, yellow)",
         "UNI": "#E2F4C5"}
options = {"ents": ["SUB", "DEC", "ROU", "TIM","UNI", "MEAS",'RTIM'], "colors": colors }

# Render NER visualization for reference annotations
html = displacy.render(doc_pred3, style="ent", options = options, page = True)
output_path = Path("model3_pred.html")
output_path.open("w", encoding="utf-8").write(html)