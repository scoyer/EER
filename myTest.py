from utils.config import *
from models.EER import *
import os

directory = args['path'].split("/")
OT = True if 'OT' in directory[1] else False
DS = directory[1].split('DS')[1].split('HDD')[0]
HDD = directory[1].split('HDD')[1].split('BSZ')[0]
BSZ =  int(directory[1].split('BSZ')[1].split('DR')[0])
B = directory[1].split('RS')[1].split('BLEU')[0]

if DS=='kvr': 
    from utils.utils_Ent_kvr import *
elif DS=='cam':
    from utils.utils_Ent_cam import *
else: 
    print("You need to provide the --dataset information")

train, dev, test, vocab, vocab_attribute, max_resp_len = prepare_data_seq(batch_size=BSZ, OOVTest=OT)

model = EER(DS, vocab, vocab_attribute, max_resp_len, int(HDD), int(HDD), lr=0.0,  dropout=0.0, path=args['path'], B=int(B), share_embedding=False)

acc_test = model.evaluate(test, 1e7)
