import nltk

ex = " Hello sir, How are you?. I am fine mam. Ok now let's see you report. Ok!. Sir your health is ok now but i recomment you to meet me twice a week on wednesday and sunday. Ok doctor i will take an appointment at 10 am for those two days. Ok sir thank you. come on 25th january. Thank you mam"

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
sent = preprocess(ex)

words = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'week', 'month']
str = ''
i = 0
while i < len(sent):
    if sent[i][0] in words:
        j = i - 1
        while (j >= 0 and sent[j][1] != 'VB'):
            str = sent[j][0] + ' ' + str
            j = j - 1
        str = sent[j][0] + ' ' + str
        while (i < len(sent) and sent[i][1] != '.'):
            str = str + sent[i][0] + ' '
            i = i + 1
        print(str)
        str = ''

    elif sent[i][1] == 'CD':
        j = i - 1
        while (j >= 0 and sent[j][1] != 'VB'):
            str = sent[j][0] + ' ' + str
            j = j - 1
        str = sent[j][0] + ' ' + str
        while (i < len(sent) and sent[i][1] != '.'):
            str = str + sent[i][0] + ' '
            i = i + 1
        print(str)
        str = ''
    i = i + 1
