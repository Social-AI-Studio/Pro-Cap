import torch
import numpy as np
import random

import config
import os
from train import train_for_epoch
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)
    
    
    # Create tokenizer
    constructor='build_baseline'
    if opt.MODEL=='pbm':
        from dataset import Multimodal_Data
        import pbm
        
        train_set=Multimodal_Data(opt,opt.DATASET,'train')
        test_set=Multimodal_Data(opt,opt.DATASET,'test')
        
        max_length=opt.LENGTH+opt.CAP_LENGTH
        #for one example, default 50
        #default, meme text, caption plus template
        """
        basically, length for one example: meme_text and caption
        """
        if opt.ASK_CAP!='':
            num_ask_cap=len(opt.ASK_CAP.split(','))
            print ('Number of asking captions',num_ask_cap)
            all_cap_len=opt.CAP_LENGTH * num_ask_cap#default, 12*5=60
            max_length+=all_cap_len
        if opt.NUM_MEME_CAP>0:
            num_meme_cap=opt.NUM_MEME_CAP
            max_length+=num_meme_cap*opt.CAP_LENGTH#default, 12*x
            print ('Number of meme aware captions',num_meme_cap)
        if opt.USE_DEMO:
            max_length*=(opt.NUM_SAMPLE*opt.NUM_LABELS+1)
            
        label_words=[opt.POS_WORD,opt.NEG_WORD]
        model=getattr(pbm,constructor)(label_words,max_length).cuda()
        
    train_loader=DataLoader(train_set,
                            opt.BATCH_SIZE,
                            shuffle=True,
                            num_workers=2)
    test_loader=DataLoader(test_set,
                           opt.BATCH_SIZE,
                           shuffle=False,
                           num_workers=2)
    train_for_epoch(opt,model,train_loader,test_loader)
    
    exit(0)
    