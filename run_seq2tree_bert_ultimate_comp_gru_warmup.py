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


def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file



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



batch_size = 16

hidden_size = config.hidden_size
embedding_size = config.embedding_size

learning_rate_embedding = 5e-5
learning_rate = 5e-5

weight_decay = 1e-5
beam_size = 5
n_layers = 2


#num_list_text = ['NUM']
num_list_text = []
for d in range(config.quantity_num):
    num_list_text.append('NUM'+str(d))


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
    tokenizer.add_special_tokens({'additional_special_tokens':num_list_text})


vocab_size = len(tokenizer)
data = load_raw_data("data/Math_23K.json")
pairs, generate_nums, copy_nums = transfer_num(data)


temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(config.ori_path, config.prefix, data, pairs)
best_acc_fold = []


pairs_trained = train_fold
pairs_tested = test_fold
pairs_validated = valid_fold
    



    

"""
for fold_t in range(5):
if fold_t == fold:
    pairs_tested += fold_pairs[fold_t]
else:
    pairs_trained += fold_pairs[fold_t]
"""
input_lang, output_lang, train_pairs, (valid_pairs, test_pairs) = prepare_data(tokenizer, pairs_trained, [pairs_validated, pairs_tested], 5, generate_nums,
                                                            copy_nums, tree=True)
"""
train_pairs:(input_cell, len(input_cell), output_cell, len(output_cell), pair[2], pair[3], num_stack)
"""


# Initialize models
from transformers import AutoModel
embedding = AutoModel.from_pretrained(config.MODEL_NAME)
embedding.resize_token_embeddings(vocab_size)
encoder = EncoderSeq_Gru(input_size=input_lang.n_words, vocab_size=vocab_size, hidden_size=hidden_size,
                n_layers=n_layers, embedding_size=embedding_size)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                    embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

# the embedding layer is  only for generated number embeddings, operators, and paddings
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

embedding_optimizer = torch.optim.AdamW(embedding.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)


encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
predict_scheduler = torch.optim.lr_scheduler.MultiStepLR(predict_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
generate_scheduler = torch.optim.lr_scheduler.MultiStepLR(generate_optimizer, milestones=[config.n_epochs//3], gamma=0.1)
merge_scheduler = torch.optim.lr_scheduler.MultiStepLR(merge_optimizer, milestones=[config.n_epochs//3], gamma=0.1)

embedding_scheduler = torch.optim.lr_scheduler.MultiStepLR(embedding_optimizer, milestones=[config.n_epochs//3], gamma=0.1)

encoder_warmup_scheduler = warmup.UntunedLinearWarmup(encoder_optimizer)
encoder_warmup_scheduler.last_step = -1 # initialize the step counter

predict_warmup_scheduler = warmup.UntunedLinearWarmup(predict_optimizer)
predict_warmup_scheduler.last_step = -1

generate_warmup_scheduler = warmup.UntunedLinearWarmup(generate_optimizer)
generate_warmup_scheduler.last_step = -1

merge_warmup_scheduler = warmup.UntunedLinearWarmup(merge_optimizer)
merge_warmup_scheduler.last_step = -1

embedding_warmup_scheduler = warmup.UntunedLinearWarmup(embedding_optimizer)
embedding_warmup_scheduler.last_step = -1

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    embedding.cuda()
generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(1, config.n_epochs+1):
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()

    all_len   = len(input_lengths)
    range_len = range(all_len)

    kl_loss_total = 0
    loss_total_prue = 0

    
    for idx in tqdm(range_len):#range_len:
        encoder_scheduler.step(epoch-1)
        predict_scheduler.step(epoch-1)
        generate_scheduler.step(epoch-1)
        merge_scheduler.step(epoch-1)
        embedding_scheduler.step(epoch-1)

        encoder_warmup_scheduler.dampen()
        predict_warmup_scheduler.dampen()
        generate_warmup_scheduler.dampen()
        merge_warmup_scheduler.dampen()
        embedding_warmup_scheduler.dampen()
        
        loss_prue, vae_kl = train_tree_gru(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], 
                generate_num_ids, encoder, predict, generate, merge, embedding,
                encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, embedding_optimizer,
                output_lang, num_pos_batches[idx], is_train = True)
        kl_loss_total += vae_kl
        loss_total_prue += loss_prue
       


    L = len(input_lengths)
    print("training time  loss:", time_since(time.time() - start))
    print("loss:{} vae_loss:{} loss type:{}".format(loss_total_prue / L, kl_loss_total / L, config.RDloss))
    print("--------------------------------")
    if (epoch-1) % config.test_interval == 0 or (epoch-1) > config.n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        #global_unstructured_flag(parameters_to_prune, config.is_prune2test)

        for test_batch in tqdm(valid_pairs):
            test_res = evaluate_tree_gru(test_batch[0], test_batch[1], generate_num_ids, 
                                     encoder, predict, generate, merge, embedding,
                                     output_lang, test_batch[5], beam_size=beam_size)
                                    
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1

        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))

        print("dropout:{} contra_weight:{} is_mask:{} is_em_dropout:{} is_prune2test:{} prunePercent:{} embedding_size:{} hidden_size:{} warm_up_strategy:{} model_name:{} learning_rate_embedding:{} learning_rate :{}".format(config.dropout, config.contra_weight, config.is_mask, config.is_em_dropout, config.is_prune2test, config.prunePercent, config.embedding_size, config.hidden_size, config.warm_up_stratege, config.MODEL_NAME, learning_rate_embedding, learning_rate ))
        print("------------------------------------------------------")
        if epoch == config.n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))

value_ac = 0
equation_ac = 0
eval_total = 0
start = time.time()



for test_batch in tqdm(test_pairs):
    test_res = evaluate_tree_gru(test_batch[0], test_batch[1], generate_num_ids, 
                             encoder, predict, generate, merge, embedding,
                             output_lang, test_batch[5], beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    if val_ac:
        value_ac += 1
    if equ_ac:
        equation_ac += 1
    eval_total += 1
print(equation_ac, value_ac, eval_total)
print("valid_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
print("valid time", time_since(time.time() - start))
print("------------------------------------------------------")
