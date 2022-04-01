import pickle

from copy import deepcopy
from src.models_vae_divide import *
from src.train_and_evaluate_divide_vae import *

def compute_midfix_expression(pre_fix, output_lang):
    st = list()
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()    # 倒着遍历
    index=0
    output = list()   # 保存运算顺序
    for token in pre_fix:
        if token[0]=='N' or token[0]=='C':      # 将数字加入堆栈
            st.append(token)
        elif output_lang.word2op_num[token]==1 and len(st) > 0:#p == "+" and len(st) > 1:
            output.append(token)
            a = st.pop()   # 先弹出来的是靠左边结点
            output.append(a)
            st.append("V_"+str(index))
            index = index + 1
        elif output_lang.word2op_num[token]==2 and len(st) > 1:
            output.append(token)
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            st.append("V_"+str(index))
            index = index + 1
        elif output_lang.word2op_num[token]==3 and len(st) > 2:
            output.append(token)
            a = st.pop()
            b = st.pop()
            c = st.pop()
            output.append(a)
            output.append(b)
            output.append(c)
            st.append("V_"+str(index))
            index = index + 1
        else:
            return None
    if len(st) == 1:
        if st[0][0]=='N' or st[0][0]=='C':
           return st
        else:
           return output#st.pop()
    return None

op_dict = {0: 'g_equal', 1: 'g_double', 2: 'g_half', 3: 'g_add', 4: 'g_minus',
          5: 'g_sin', 6: 'g_cos', 7: 'g_tan', 8: 'g_asin', 9: 'g_acos',
          10: 'gougu_add', 11: 'gougu_minus', 12: 'g_bili',
          13: 'g_mul', 14: 'g_divide', 15: 'cal_circle_area', 16: 'cal_circle_perimeter', 17: 'cal_cone'}

def compute_prefix_expression(pre_fix):
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
            st.append("V_"+str(index))
            index = index + 1
        elif p == "*" and len(st) > 1:
            output.append("g_mul")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            st.append("V_"+str(index))
            index = index + 1
            #st.append(a * b)
        elif p == "/" and len(st) > 1:
            output.append("g_divide")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            st.append("V_"+str(index))
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
            st.append("V_"+str(index))
            index = index + 1
            #st.append(a - b)
        elif p == "^" and len(st) > 1:
            output.append("g_exp")
            a = st.pop()
            b = st.pop()
            output.append(a)
            output.append(b)
            st.append("V_"+str(index))
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


program_list = []
def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""
    return data

data = load_raw_data("data/Math_23K.json")
pairs, generate_nums, copy_nums = transfer_num(data)


temp_pairs = []
for p in pairs:
    temp_pairs.append((from_infix_to_prefix(p[1])))
pairs = temp_pairs
for data in pairs:
    print("data:{}".format(data))
    output= compute_prefix_expression(data)
    print("output:{}".format(output))


















op_num = {}
call_op = {}
for op in op_list:
            call_op[op] = eval('operators.{}'.format(op))
            op_num[op] = call_op[op].__code__.co_argcount

id_list=[]
token_list0 = []
with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            index = 0
            for sample in dataset:
                #print("|||||||||||||||||||||||||||||sample:{}=============================================================================".format(sample))
                #print("manual_program:{}".format(sample["manual_program"]))
                program =sample["manual_program"]  
                if index == 49:
                   print("source text:{}".format(sample['token_list']))
                   print("program:{}".format(sample["manual_program"]))
                   program = ['g_minus' ,'C_3' , 'N_1', 'g_minus', 'C_3', 'V_0', 'g_minus', 'V_1', 'N_0', 'g_minus', 'C_3', 'V_2' ]
                program_list.append(program)
                id_list.append(sample["id"])
                token_list0.append(sample['token_list'])
                index = index + 1
                #print("tokens:{}".format(sample['token_list']))

print(op_list)

f = open('output_lang.pk1', 'rb')
output_lang = pickle.load(f)
f.close()
print("output_lang:{}".format(output_lang.word2op_num))
token_list = []
program_list_new = []
index = 0
for Tokens, program, id in zip(token_list0, program_list, id_list):
    mini_token=[]  # 包含参数命令
    tokens=[]      # 参数命令 list
    #print("tokens:{} program:{}".format(tokens, program))
    if "N_11" in program:
        print("tokens:{} program:{} id:{}".format(Tokens, program, id))

    if "N_21" in program:
        program[program.index("N_21")] = "N_1"
        print("tokens:{} program:{} id:{}".format(Tokens, program, id))
    for i,token in enumerate(program):
        
        if token in  op_list:
            mini_token=[]
            mini_token.append(token)
        else:
            if(token=='V_N_0'):
               token='V_0'
               program[i]='V_0'
            mini_token.append(token)
            #print("i+1:{} len(program)-1:{}".format(i+1, len(program)-1))
            if i == (len(program)-1) or program[i+1] in op_list:
                #print("mini_token:{}\n".format(mini_token))

                tokens.append(mini_token)
    if index == 49:
        print("id:{}".format(id))
    index = index + 1
    token_list.append(tokens)
    program_list_new.append(program)
    

# 自顶向下建树
def dfs(now, program, vis):
    
    for i,token in enumerate(now[1]):        # 遍历它的孩子结点
            
            # print("token:{} i:{}\n program:{}".format(token, i, program))
            if token[0]=='V':                    # 如果是孩子结点就建树
                #print("token[2:]:{}".format(token[2:]))
                child_number = int(token[2:])     # 得到子结点下标值
                child = [program[child_number][0], program[child_number][1:], token]
                #print("===================================================================2")
                now[1][i]=child                   # 更新孩子结点
                # print("===================================================================1")
                # print("now[1][i]:{}".format(now[1][i]))
                dfs(now[1][i], program, vis)
                # print("now[1][i]:{}".format(now[1][i]))
                # print("===================================================================2")

root_list=[]
# 这里逆序遍历, 从根开始遍历
index = 0
for  program, tokens in zip(program_list, token_list):
    print("tokens:{}".format(tokens))
    print("program:{}".format(program))
    root = [tokens[-1][0], tokens[-1][1:], None]    # 保存所有的树结构
    vis = list()
    dfs(root, tokens, vis)
    print("root:{}".format(root))
    #print("root:{} program:{}".format(root, program))

    index = index+1
    root_list.append(root)



def Postorder_traversal(root,  post_list):
    if isinstance (root, str)==True:
        post_list.append(root)
        return

       
    for node in root[1]:
        Postorder_traversal(node,  post_list)
    post_list.append(root[0])

def Preorder_traversal(root,  post_list):
    if isinstance (root, str)==True:
        post_list.append(root)
        return
    #print("root[0]:{}".format(root[0]))
    post_list.append(root[0])
    for node in root[1]:
        Preorder_traversal(node,  post_list)
    


# 得到后序遍历, 先遍历叶子结点, 再遍历根结点
post_list = []


index = 0
for program, root in zip(program_list,root_list):
   post=[]
    
#    print("root:{} program:{}".format(root, program))
   Preorder_traversal(root, post)
   #print("pre:{}".format(post))
   #print("=================================================end")
   print("=================================================start index:{}".format(index))
   post_list.append(post)
   mid = compute_midfix_expression(post, output_lang)
   print("root:{}".format(root))
   print("pre:{}".format(post))
   print("mid:{}".format(mid))
#    print("program:{}".format(program))
#    if index not in [66,86, 87, 88]:
#       assert mid == program
  
   print("=================================================end")
   index=index+1
   

data_list =[]
# for post, id, program, root in zip(post_list, id_list, program_list, root_list):
#     print("=========================================start")
#     print("post:{}".format(post))
#     print("program:{}".format(program))
#     print("root:{}".format(root))
#     print("=========================================end")


sample_post=[]
with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            
            for sample, post, program in zip(dataset, post_list, program_list_new):
                sample["pre"] = post
                #print("manual_program2:{}".format(program))
                sample["manual_program"] = program
                sample_post.append(sample)
                
                # program_list.append(sample["manual_program"])
                # id_list.append(sample["id"])

file_path_post = "data/GeoQA3/test_pre.pk"
# if not exit(file_path_post):
#     mkdir(file_path_post)


with open(file_path_post, 'wb') as f:
	pickle.dump(sample_post, f)
