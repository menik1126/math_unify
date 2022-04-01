# coding: utf-8
# from src.train_and_evaluate_prune import *
# from src.models_prune import *

from src.models_vae_divide import *
from src.train_and_evaluate_divide_vae import *

import time
import torch.optim

from src.expressions_transfer import *
from tqdm import tqdm

from src import config
#import torch.nn.utils.prune as prune
#from src.prune_method import *
import pytorch_warmup as warmup
from ManualProgram.eval_equ import Equations

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def compute_prefix_expression(pre_fix, generate_nums):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    index=0
    output = list()
    for p in pre_fix:
        if p not in operators:
            # pos = re.search("\d+\(", p)
            # if pos:
            #     st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            # elif p[-1] == "%":
            #     st.append(float(p[:-1]) / 100)
            # else:
            #     st.append(eval(p))
            st.append(p)
        elif p == "+" and len(st) > 1:
            output.append("g_add")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            #st.append(a + b)
            label = "V_"+str(index)
            if label not in generate_nums:
                generate_nums.append(label)
            st.append(label)
            index = index + 1
        elif p == "*" and len(st) > 1:
            output.append("g_mul")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            label = "V_"+str(index)
            if label not in generate_nums:
                generate_nums.append(label)
            st.append(label)
            index = index + 1
            #st.append(a * b)
        elif p == "/" and len(st) > 1:
            output.append("g_divide")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            label = "V_"+str(index)
            if label not in generate_nums:
                generate_nums.append(label)
            st.append(label)
            if b == 0:
                return None
            index = index + 1
            #st.append(a / b)
        elif p == "-" and len(st) > 1:
            output.append("g_minus")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            label = "V_"+str(index)
            if label not in generate_nums:
                generate_nums.append(label)
            st.append(label)
            index = index + 1
            #st.append(a - b)
        elif p == "^" and len(st) > 1:
            output.append("g_exp")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            label = "V_"+str(index)
            if label not in generate_nums:
                generate_nums.append(label)
            st.append(label)
            index = index + 1
            # if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
            #     return None
            #st.append(a ** b)
        else:
            return None

    if len(st) == 1:
        if st[0][0]!="V" or st[0][0].isdigit():
            return st
        else:
            return output#st.pop()
    return None



def get_train_test_fold(ori_path, prefix, data, pairs):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []


    for item, pair in zip(data, pairs):
        pair = list(pair)
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

def out_program_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    cnt = 0
    length = len(test)
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            # if idx[0] == "N":
            #     if int(idx[1:]) >= len(num_list):
            #         #print("here=================")
            #         return None
            #     res.append(num_list[int(idx[1:])])
            # else:
            res.append(idx)
        else:
                
                #print("token:{}".format(token))
                
                pos_list = num_stack.pop()
                c = num_list[pos_list[0]]
                res.append(c)
          
    return res

def compute_program_tree_result(test_res, test_tar, output_lang, num_list, num_stack, _equ, gt_ans, gt):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_program_list(test_res, output_lang, num_list)
    
    if  test[-1]=='EOS':
        test = test[:-1]
        #print("hypo:{}".format(test))

    #print("test:{}".format(test))
    tar = out_program_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    tar = tar[:-1]
    #print("tar:{}".format(tar))
    ans = _equ.excuate_equation(tar, num_list)
    
    # p = gt_ans
    # pos = re.search("\d+\(", p)
    # print("gt_ans:{}".format(p))
    # print("gt_equation:{}".format(gt))
    # print("num_list:{}".format(num_list))
    # if pos:
    #     gt_ans = eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:])
    # elif p[-1] == "%":
    #     gt_ans = float(p[:-1]) / 100
    # else:
    #     gt_ans =eval(p)
    # print("gt_ans:{} ans:{}".format(gt_ans, ans[-1]))
    # assert abs(ans[-1] - float(gt_ans))<1e-4
    #print("test_tar:{}".format(test_tar))
    
    # print("tar:{}".format(tar))
    # print("tar_ans:{}".format(_equ.excuate_equation(tar, num_list)))

    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(_equ.excuate_equation(test, num_list) - ans[-1]) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


batch_size = 16

hidden_size =config.hidden_size# 512

learning_rate = 5e-5
weight_decay = 1e-5
beam_size = 5
n_layers = 2


#num_list_text = ['NUM']
num_list_text = []
for d in range(config.quantity_num):
    num_list_text.append('NUM'+str(d))

embedding_size = config.embedding_size
if config.MODEL_NAME=='roberta':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif config.MODEL_NAME=='roberta-large':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("./src/chinese_roberta_large/vocab.txt")
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})
elif config.MODEL_NAME =='xml-roberta':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')#("./src/chinese_roberta/vocab.txt")#, additional_special_tokens = num_list_text )
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif config.MODEL_NAME =='xml-roberta-base':
    from transformers import XLMRobertaTokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')#("./src/chinese_roberta/vocab.txt")#, additional_special_tokens = num_list_text )
    tokenizer.additional_special_tokens = tokenizer.additional_special_tokens + num_list_text
elif config.MODEL_NAME =='bert-base-chinese':
    from transformers import AutoTokenizer
    print("model name:{}".format(config.MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


vocab_size = len(tokenizer)
data = load_raw_data("data/Math_23K.json")
pairs, generate_nums, copy_nums = transfer_num(data)


temp_pairs = []
for p in pairs:
    pre = from_infix_to_prefix(p[1])
    # 加入v为generate的num
    mid = compute_prefix_expression(pre, generate_nums)
    #print("mid:{}".format(mid))
    if mid is None:
        print("p[1]:{}".format(p[1]))
        print("pre:{}".format(pre))
        print("mid:{}".format(mid))
        mid = [ 'g_add', 'N_0', 'N_1']

    temp_pairs.append((p[0], mid, p[2], p[3], p[4], p[1]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(config.ori_path, config.prefix, data, pairs)
best_acc_fold = []


pairs_trained = train_fold
pairs_tested = test_fold
pairs_validated = valid_fold
    


_equ = Equations()
    

"""
for fold_t in range(5):
if fold_t == fold:
    pairs_tested += fold_pairs[fold_t]
else:
    pairs_trained += fold_pairs[fold_t]
"""
input_lang, output_lang, train_pairs, (valid_pairs, test_pairs) = prepare_data(tokenizer, pairs_trained, [pairs_validated, pairs_tested], 5, generate_nums,
                                                            copy_nums, tree=False)
"""
train_pairs:(input_cell, len(input_cell), output_cell, len(output_cell), pair[2], pair[3], num_stack)
"""
print("output_lang word:{}".format(output_lang.word2index))


# Initialize models
encoder = EncoderSeq_noVAE(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                n_layers=n_layers)
decoder = AttnDecoderRNN(hidden_size=hidden_size, embedding_size=embedding_size, input_size=output_lang.n_words, output_size=output_lang.n_words, n_layers=n_layers)
# the embedding layer is  only for generated number embeddings, operators, and paddings
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[config.n_epochs//3], gamma=0.1)

encoder_warmup_scheduler = warmup.UntunedLinearWarmup(encoder_optimizer)
encoder_warmup_scheduler.last_step = -1 # initialize the step counter

decoder_warmup_scheduler = warmup.UntunedLinearWarmup(decoder_optimizer)
decoder_warmup_scheduler.last_step = -1






# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()


        

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])
print("generate_nums:{}".format(generate_nums))
for epoch in range(1, config.n_epochs+1):
    
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()

    all_len   = len(input_lengths)
    range_len = range(all_len)
             
    loss_total = 0
    for idx in tqdm(range_len):#range_len:
        encoder_scheduler.step(epoch-1)
        decoder_scheduler.step(epoch-1)

        encoder_warmup_scheduler.dampen()
        decoder_warmup_scheduler.dampen()

        loss = train_attn_prog(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            nums_batches[idx], num_stack_batches[idx], copy_nums, generate_num_ids,
            encoder, decoder, encoder_optimizer, decoder_optimizer, 
            output_lang, num_pos_batches[idx], beam_size=beam_size)
        
        loss_total += loss
       


    L = len(input_lengths)
    print("training time  loss:", time_since(time.time() - start))
    print("loss:{} ".format(loss_total / L))
    print("--------------------------------")
    if (epoch-1) % config.test_interval == 0 or (epoch-1) > config.n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        #global_unstructured_flag(parameters_to_prune, config.is_prune2test)

        for test_batch in tqdm(valid_pairs):
            # (input_seq, out_seq, nums, num_pos)
            # train_pairs:(input_cell, len(input_cell), output_cell, len(output_cell), pair[2], pair[3], num_stack)

            # evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
            #       beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
        
            attn_beams = evaluate_attn_prog(test_batch[0], test_batch[1], 
                                        copy_nums, generate_num_ids, 
                                        encoder, decoder,
                                        output_lang,
                                        beam_size=beam_size)
            # val_ac = False
            # equ_ac = False
            # for attn_beam in attn_beams:
            #     # 选中第一个退出
            #     if val_ac==True:
            #             break
            #     test_res = attn_beam.all_output
                #print("len of test_batch[6]:{}".format(len(test_batch[6])))

                
                #print("test:{} test_batch[4]:{}".format(hypo, test_batch[4])) val_ac, equ_ac, _, _  = 
            test_res = attn_beams[0].all_output
            val_ac, equ_ac, _, _ = compute_program_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6], _equ, test_batch[7], test_batch[8])
                
               
                #print("res:{}".format(res))
                #choice_nums=test_batch[7]
                #print("choice_nums:{}".format(choice_nums))
               
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1

        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))

        print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{} warm_up_strategy:{} model_name:{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size, config.warm_up_stratege, config.MODEL_NAME))
        # print("------------------------------------------------------")
        # torch.save(encoder.state_dict(), "models/encoder")
        # torch.save(predict.state_dict(), "models/predict")
        # torch.save(generate.state_dict(), "models/generate")
        # torch.save(merge.state_dict(), "models/merge")






value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()
for test_batch in tqdm(test_pairs):
            choice = None
            attn_beams = evaluate_attn_prog(test_batch[0], test_batch[1], 
                                        copy_nums, generate_num_ids, 
                                        encoder, decoder,
                                        output_lang,
                                        beam_size=beam_size)
            # val_ac = False
            # equ_ac = False
            # for attn_beam in attn_beams:
            #     # 选中第一个退出
            #     if val_ac==True:
            #             break
            test_res = attn_beams[0].all_output
            val_ac, equ_ac, _, _ = compute_program_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6], _equ, test_batch[7], test_batch[8])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")