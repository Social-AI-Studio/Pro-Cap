import os
import json
import pickle as pkl
import numpy as np
import torch
import utils
from tqdm import tqdm
import config
import random

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Multimodal_Data():
    #mem, off, harm
    def __init__(self,opt,dataset,mode='train'):
        super(Multimodal_Data,self).__init__()
        self.opt=opt
        self.mode=mode
        
        self.num_ans=self.opt.NUM_LABELS
        self.num_sample=self.opt.NUM_SAMPLE
        if len(self.opt.ASK_CAP.split(','))>=1:
            self.ask_cap=True
        else:
            self.ask_cap=False
        self.add_ent=self.opt.ADD_ENT
        self.add_dem=self.opt.ADD_DEM
        self.num_meme_cap=self.opt.NUM_MEME_CAP
        print ('Adding exntity information?',self.add_ent)
        print ('Adding demographic information?',self.add_dem)
        
        self.label_mapping_word={0:self.opt.POS_WORD,
                                 1:self.opt.NEG_WORD}
        self.template="*<s>**sent_0*.*_It_was*label_**</s>*"
        
        self.template_list=self.template.split('*')
        print('Template:', self.template)
        print('Template list:',self.template_list)
        
        self.support_examples=self.load_entries('train')
        print ('Length of supporting example:',len(self.support_examples))
        self.entries=self.load_entries(mode)
        if self.opt.DEBUG:
            self.entries=self.entries[:128]
        self.prepare_exp()
        print ('The length of the dataset for:',mode,'is:',len(self.entries))

    def load_entries(self,mode):
        #print ('Loading data from:',self.dataset)
        #only in training mode, in few-shot setting the loading will be different
        path=os.path.join(self.opt.DATA,
                          'domain_splits',
                          self.opt.DATASET+'_'+mode+'.json')
        data=read_json(path)
        if self.opt.CAP_TYPE=='caption':
            cap_path=os.path.join(self.opt.CAPTION_PATH,
                                  self.opt.DATASET+'_'+self.opt.PRETRAIN_DATA,
                                  self.opt.IMG_VERSION+'_captions.pkl')
        elif self.opt.CAP_TYPE=='vqa':
            cap_path=os.path.join('../../Ask-Captions/Captions',
                                  self.opt.DATASET,
                                  mode+'_generic.pkl')
            if self.opt.ASK_CAP!='':
                questions=self.opt.ASK_CAP.split(',')
                result_files={q:load_pkl(os.path.join(
                    '../../Ask-Captions/'+self.opt.LONG+'Captions',
                    self.opt.DATASET,
                    mode+'_'+q+'.pkl')) 
                              for q in questions}
                print (len(result_files))
                valid=['valid_person','valid_animal']
                for v in valid:
                    result_files[v]=load_pkl(os.path.join(
                        '../../Ask-Captions/'+self.opt.LONG+'Captions',
                        self.opt.DATASET,
                        mode+'_'+v+'.pkl'))
        
        captions=load_pkl(cap_path)
        entries=[]
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            if self.opt.CAP_TYPE=='caption':
                cap=captions[img.split('.')[0]][:-1]#remove the punctuation in the end
            elif self.opt.CAP_TYPE=='vqa' and self.ask_cap:
                cap=captions[img]
                
                ext=[]
                person_flag=True
                animal_flag=True
                person=result_files['valid_person'][row['img']].lower()
                if person.startswith('no'):
                    person_flag=False
                animal=result_files['valid_animal'][row['img']].lower()
                if animal.startswith('no'):
                    animal_flag=False
                for q in questions:
                    if person_flag==False and q in ['race','gender',
                                                    'country','valid_disable']:
                        continue
                    if animal_flag==False and q=='animal':
                        continue
                    if q in ['valid_person','valid_animal']:
                        continue
                    info=result_files[q][row['img']]
                    if q=='valid_disable':
                        if info.startswith('no'):
                            continue
                        else:
                            ext.append('there is a disabled person')
                    else:
                        ext.append(info)
                if self.num_meme_cap>0:
                    pnp_cap_path=os.path.join('../../Ask-Captions/pnp-captions',
                                              self.opt.DATASET,img+'.json')
                    if os.path.exists(pnp_cap_path):
                        caps=read_json(pnp_cap_path)
                        ext.extend(caps[:self.num_meme_cap])
                    else:
                        ext.extend([cap]*self.num_meme_cap)      
                #print(ext[-10:])
                ext=' . '.join(ext)
                cap=cap+' . '+ext
            #no asking captions
            elif self.opt.CAP_TYPE=='vqa':
                cap=captions[img]
                ext=[]
                if self.num_meme_cap>0:
                    pnp_cap_path=os.path.join('../../Ask-Captions/pnp-captions',
                                              self.opt.DATASET,img+'.json')
                    if os.path.exists(pnp_cap_path):
                        caps=read_json(pnp_cap_path)
                        ext.extend(caps[:self.num_meme_cap])
                    else:
                        ext.extend([cap]*self.num_meme_cap)      
                #print(ext[-10:])
                ext=' . '.join(ext)
                cap=cap+' . '+ext
                
            sent=row['clean_sent']
            #whether using external knowledge
            if self.add_ent:
                cap=cap+' . '+row['entity']+' . '
            if self.add_dem:
                cap=cap+' . '+row['race']+' . '
            entry={
                'cap':cap.strip(),#generic_cap, ask_cap, meme_aware_cap, external knowledge
                'meme_text':sent,
                'label':label,
                'img':img
            }
            entries.append(entry)
        return entries
    
    def enc(self,text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def prepare_exp(self):
        ###add sampling
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                context_indices = [support_idx for support_idx in support_indices
                                   if support_idx != query_idx or self.mode != "train"]
                #available indexes for supporting examples
                self.example_idx.append((query_idx, context_indices, sample_idx))

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        num_labels=self.opt.NUM_LABELS
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
        if num_labels == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []
        """
        # Sampling strategy from LM-BFF
        if self.opt.DEBUG:
            print ('Number of context examples available:',len(context_examples))
        """
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]['label']
            if num_labels == 1:
                # Regression
                #No implementation currently
                label = '0' if\
                float(label) <= median_mapping[self.args.task_name] else '1'
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        
        assert len(selection) > 0
        return selection
    
    def process_prompt(self, examples):
        prompt_arch=' It was '
        concat_sent=[]
        
        for segment_id, ent in enumerate(examples):
            if segment_id==0:
                #implementation for the querying example
                temp=prompt_arch+'<mask> . '
            else:
                label_word=self.label_mapping_word[ent['label']]
                temp=prompt_arch+label_word+' . '
            #put the temp in the middle to avoid ignorance
            whole_sent=ent['meme_text']+' . '+temp+ent['cap']
            concat_sent.append(whole_sent)
            if segment_id==0:
                test_text=ent['meme_text']+' . '+ent['cap']
        return concat_sent,test_text

                
    def __getitem__(self,index):
        #query item
        entry=self.entries[index]
        #bootstrap_idx --> sample_idx
        query_idx, context_indices, bootstrap_idx = self.example_idx[index]
        #one example from each class
        supports = self.select_context(
            [self.support_examples[i] for i in context_indices])
        exps=[]
        exps.append(entry)
        exps.extend(supports)
        concate_sent,test_text = self.process_prompt(exps)
        prompt_texts=' . </s> '.join(concate_sent)
        
        vid=entry['img']
        #label=torch.tensor(self.label_mapping_id[entry['label']])
        label=torch.tensor(entry['label'])
        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        target[label]=1.0
        """
        print ('Test sent:')
        print ('\t',concate_sent[0]+' . </s> ')
        print ('Prompt text:')
        print ('\t',prompt_texts)
        """
        batch={
            'img':vid,
            'target':target,
            'test_all_text':concate_sent[0]+' . </s> ',
            'test_text':test_text,
            'prompt_all_text':prompt_texts,
            'label':label
        }
        return batch
        
    def __len__(self):
        return len(self.entries)
    
