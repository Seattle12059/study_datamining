with open( 'train.txt','r',encoding='utf-8') as f:
    for i in range(9):  # 读取前9行
        line = f.readline()
        if not line:  # 如果行为空（文件行数少于6行），提前结束
            break
        print(line.strip())  # 去掉行尾的换行符并打印



#%%
def read(data_path):
    data=['label'+'\t'+'text_a\n']
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        lines=f.readlines()
        # 三行为一条记录
        for i in range(int(len(lines)/3)):
            # 读取第一行为内容
            word = lines[i*3].strip('\n')
            # 读取第三行为标签
            label = lines[i*3+2].strip('\n')
            data.append(label+'\t'+word+'\n')
            i=i+1
    return data
with open('formated_train.txt','w') as f:
    f.writelines(read('train.txt'))

with open('formated_test.txt','w') as f:
    f.writelines(read('test.txt'))


#%%
with open( 'formated_train.txt','r') as f:
    for i in range(9):  # 读取前9行
        line = f.readline()
        if not line:  # 如果行为空（文件行数少于6行），提前结束
            break
        print(line.strip())  # 去掉行尾的换行符并打印

with open( 'formated_test.txt','r') as f:
    for i in range(9):  # 读取前9行
        line = f.readline()
        if not line:  # 如果行为空（文件行数少于6行），提前结束
            break
        print(line.strip())  # 去掉行尾的换行符并打印


#%%
from paddlenlp.datasets import load_dataset

def read(data_path):
    with open(data_path, 'r') as f:
        # 跳过列名
        next(f)
        for line in f:
            label,  word= line.strip('\n').split('\t')
            yield {'text': word, 'label': label}

# data_path为read()方法的参数
train_ds = load_dataset(read, data_path='formated_train.txt',lazy=False)
test_ds = load_dataset(read, data_path='formated_test.txt',lazy=False)
dev_ds = load_dataset(read, data_path='formated_test.txt',lazy=False)



#%%
print(len(train_ds))
print(train_ds.label_list)
for idx in range(10):
    print(train_ds[idx])

#%%
import paddlenlp as ppnlp

# 设置想要使用模型的名称
MODEL_NAME = "ernie-1.0"
ernie_model  = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=3)


tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
#%%
