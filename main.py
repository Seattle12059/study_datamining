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
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader

# 模型运行批处理大小
batch_size = 20
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


#%%
from paddlenlp.transformers import LinearDecayWithWarmup
import paddle

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 5 #3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

#%%

import paddle.nn.functional as F
from utils import evaluate

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)
