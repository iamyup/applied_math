import glob, os.path
from collections import *
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

data_spm = []
data_ham = []

for d in range(1, 10):
    for file in glob.glob('part{:d}/*.txt'.format(d)):
        #print(file)
        with open(file, 'r', encoding='UTF-8') as txt:
            # each line make to string
            for line in txt:
                strings = line.split(' ')
                # clean up
                for string in strings:
                    # ignore space and new line
                    if string is not '\n' and string is not '' and string is not ' ':
                        # first clean-up: lower case
                        string = string.lower()
                        # second clean-up: weird characters
                        string = re.sub(r'\W+', '', string)
                        if string is not '':
                            if os.path.basename(file).startswith('spm'):
                                data_spm.append(string)
                            else:
                                data_ham.append(string)
        #print('FINISH :',file,os.path.basename(file).startswith('spm'))
        #print('spm',len(data_spm))
        #print('ham',len(data_ham))

# Make them as dictionaries of unigram
unigram_spm = defaultdict(int)
unigram_ham = defaultdict(int)
for k in data_spm:    unigram_spm[k] += 1
for k in data_ham:    unigram_ham[k] += 1

# Normalize the count into probability to appear : the sum is to 1
for key, value in unigram_spm.items():
    unigram_spm[key] = value / len(data_spm)
for key, value in unigram_ham.items():
    unigram_ham[key] = value / len(data_ham)

# Bayesian HAM-SPAM TEST for 10th folder

real_ham_spam = [] # ham : 0, spam : 1
classified = []

for file in glob.glob('part10\*.txt'.format(d)):
#for file in glob.glob('part11\*.txt'.format(d)):
    print('Checking.....',file)
    test_data_spm = [] #reset list
    test_data_ham = [] #reset list
    likelihood_spm = 0
    likelihood_ham = 0
    #print('CHECK!')
    with open(file, 'r', encoding='UTF-8') as txt:
        # each line make to string
        for line in txt:
            strings = line.split(' ')
            # clean up
            for string in strings:
                if string is not '\n' and string is not '' and string is not ' ':
                    string = string.lower()
                    string = re.sub(r'[^a-zA-Z0-9]', '', string) # for better Classification!!
                    #string = re.sub(r'\W+', '', string)
                    if string is not '':
                        ### LIKELIHOOD of HAM :
                        if string in unigram_ham:
                            likelihood_ham += np.log(unigram_ham[string])
                        else:
                            likelihood_ham += np.log(1/len(data_ham))

                        ### LIKELIHOOD of SPAM :
                        if string in unigram_spm:
                            likelihood_spm += np.log(unigram_spm[string])
                        else:
                            likelihood_spm += np.log(1/len(data_spm))

                        ## CHECK WHY
                        #print('[WORD :',string,']','[ratio :',likelihood_spm/likelihood_ham)

    #print(file,likelihood_ham,likelihood_spm)
    # Stack the results of Naive Bayesian Classifier
    if likelihood_spm > likelihood_ham :
        classified.append(1) # classify as SPAM
        #print('SPAM!!')
    else :
        classified.append(0) # classify as HAM
        #print('HAM!!')
    # Stack the real value
    if os.path.basename(file).startswith('spm'):
        real_ham_spam.append(1)  # real SPAM
    else:
        real_ham_spam.append(0)  # real HAM

    if classified[-1] != real_ham_spam[-1]:
        print('!! MISS CLASSIFIED :',file)

print('TRUE :', real_ham_spam)
print('ALARM :', classified)

label = ['ham','spam']

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['ham','spam']))
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels([item for item in label])
    plt.yticks(tick_marks)
    ax.set_yticklabels([item for item in label])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True HAM-SPAM')
    plt.xlabel('Predicted HAM-SPAM')

cm = confusion_matrix(real_ham_spam, classified)
np.set_printoptions(precision=1)
fig, ax = plt.subplots()
plot_confusion_matrix(cm)
print(' ')
print('Confusion matrix, without normalization')
print(cm)
print('NOTE : IF YOU CAN NOT SEE THE MATRIX PLOT, PLEASE CHECK VERSION OF SKIT-LEARN!')
plt.show()

''' COMMENT
1) from the confusion matrix,
  - average the spam false alarm rates : 5/242 = 2.1 %
  - average the spam hit rates : 47/49 = 95.9 %

and above results make sense.

2) Check four mistake emails and try to tell me WHY they were miss classified!

i) first, we checked "part10\9-5msg1.txt" and we found that the character "_" results in wrong judgement!
and we changed the regular expression, and we get much better matrix.
(we checked the effect of likelihood word by word.)

  - average the spam false alarm rates : 0/242 = 0.0 %
  - average the spam hit rates : 47/49 = 95.9 % (the same)

you can easily see it by using LINE 67 instead of LINE 68 in this code.

ii) we checked the last two wrong mails.
and we found two facts :

① the word used in specialized field like 'paper'(in academic) can avoid the filter. (e.g., spmsgc90.txt : mining, paper, subject)
② if the word is writen by alphabet alone like 'f i v e' can avoid the filter as well.
(e.g., spmsgc62.txt : view, "t h e h o l l a n d e r c o l l e c t i o n f i v e a r t i s t s . o n e f a m i l y")

3) Check your classifier with 10 of your own "ham" and "spam" messages - does it work? Why? Why not?

we tested our 5 spams and hams respectively. and we got
  - average the spam false alarm rates : 4/5 = 80.0 % (!!!)
  - average the spam hit rates : 5/5 = 100.0 % (good)

it does work for real spam, and not for real ham.
why? we guess that the official hams look like spam.
they use 'join','today','meet', so that can be classified to SPAM!
you can check it in part11 folder.

THANK YOU-

'''