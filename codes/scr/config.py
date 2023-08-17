import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--DATASET',type=str,default='mem')
    parser.add_argument('--MODEL',type=str,default='pbm')
    parser.add_argument('--CAP_TYPE',type=str,default='vqa')#caption
    
    #path configuration
    parser.add_argument('--DATA',
                        type=str,
                        default='/Data_Storage/Rui_Data_Space/hate-speech')
    parser.add_argument('--CAPTION_PATH',
                        type=str,
                        default='/Data_Storage/Rui_Code_Space/hate-speech/CLIP_prefix_caption')
    
    parser.add_argument('--BERT_DIM',type=int,default=768)
    parser.add_argument('--ROBERTA_DIM',type=int,default=1024)
    parser.add_argument('--NUM_LABELS',type=int,default=2)
    
    parser.add_argument('--POS_WORD',type=str,default='good')
    parser.add_argument('--NEG_WORD',type=str,default='bad')
    parser.add_argument('--MULTI_QUERY',type=bool,default=True)
    parser.add_argument('--USE_DEMO',type=bool,default=True)
    parser.add_argument('--NUM_QUERIES',type=int,default=4)
    
    #hyper parameters configuration
    parser.add_argument('--FC_DROPOUT',type=float,default=0.4) 
    parser.add_argument('--WEIGHT_DECAY',type=float,default=0.01) 
    parser.add_argument('--LR_RATE',type=float,default=1e-5) 
    parser.add_argument('--EPS',type=float,default=1e-8) 
    parser.add_argument('--BATCH_SIZE',type=int,default=16)
    parser.add_argument('--FIX_LAYERS',type=int,default=2)
    parser.add_argument('--NUM_SAMPLE', type=int, default=1)
    parser.add_argument('--NUM_MEME_CAP', type=int, default=0)
    parser.add_argument('--MID_DIM',type=int,default=512)
    
    parser.add_argument('--LENGTH',type=int,default=65)#50 plus 5 temp length
    parser.add_argument('--MODEL_NAME',type=str,default='allenai/unifiedqa-t5-small')
    
    parser.add_argument('--ASK_CAP',
                        type=str,
                        default='race,gender,country,animal,valid_disable,religion')
    parser.add_argument('--LONG',
                        type=str,
                        default='Longer-')
    parser.add_argument('--CAP_LENGTH',type=int,default=12)
    
    #optional: allenai/unifiedqa-t5-base allenai/unifiedqa-t5-large allenai/unifiedqa-t5-3b
    parser.add_argument('--PRETRAIN_DATA',type=str,default='conceptual')
    parser.add_argument('--IMG_VERSION',type=str,default='clean')
    parser.add_argument('--ADD_ENT',type=bool,default=True)
    parser.add_argument('--ADD_DEM',type=bool,default=True)
    
    parser.add_argument('--DEBUG',type=bool,default=False)
    parser.add_argument('--SAVE',type=bool,default=False)
    parser.add_argument('--SAVE_NUM',type=int,default=100)
    parser.add_argument('--EPOCHS',type=int,default=10)
    
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--CUDA_DEVICE', type=int, default=13)
    
    parser.add_argument('--WARM_UP',type=int,default=2000)
    
    args=parser.parse_args()
    return args
