import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, RobertaTokenizer

class PromptHateModel(nn.Module):
    def __init__(self,label_words,max_length=320,model_name='roberta-large'):
        super(PromptHateModel, self).__init__()
        self.roberta = RobertaForMaskedLM.from_pretrained(model_name)
        self.tokenizer=RobertaTokenizer.from_pretrained(model_name)
        self.max_length=max_length
        
        self.mask_token_id=self.tokenizer.mask_token_id
        self.label_word_list=[]
        for word in label_words:
            self.label_word_list.append(self.tokenizer._convert_token_to_id(
                self.tokenizer.tokenize(' ' + word)[0]))
        print (label_words,self.label_word_list)

    def forward_single_cap(self,tokens,attention_mask,mask_pos):
        batch_size = tokens.size(0)
        mask_pos = mask_pos.squeeze()
            
        out = self.roberta(tokens, 
                           attention_mask)
        prediction_mask_scores = out.logits[torch.arange(batch_size),
                                          mask_pos]
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        return logits
    
    def generate_input_tokens(self,sents,max_length=320):
        token_info=self.tokenizer(sents, padding='longest', 
                                  truncation=True, max_length=max_length, 
                                  return_tensors="pt")
        tokens=token_info.input_ids
        attention_mask=token_info.attention_mask
        mask_pos = [t.numpy().tolist().index(self.mask_token_id) for t in tokens]
        mask_pos=torch.LongTensor(mask_pos)
        return tokens,attention_mask,mask_pos
    
    def forward(self,all_texts):
        batch_size = len(all_texts)#
        tokens,attention_mask,mask_pos=self.generate_input_tokens(all_texts,
                                                                  self.max_length)
        logits=self.forward_single_cap(tokens.to(self.roberta.device),
                                       attention_mask.to(self.roberta.device),
                                       mask_pos.to(self.roberta.device))#B,2
        return logits
    
def build_baseline(label_words,max_length):  
    return PromptHateModel(label_words,max_length)