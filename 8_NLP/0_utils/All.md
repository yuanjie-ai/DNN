```python
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.lower()):
    from tqdm import tqdm
    with open(file_input) as ifile:
        with open(file_output, 'a') as ofile: # append
            for i in tqdm(ifile, desc='File Processing'):
                if len(re.sub('[^\u4e00-\u9fa5]+', '', str(i))) > 2: # 去除短文本
                    ofile.writelines(func(i))
```

