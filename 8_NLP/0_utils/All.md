```python
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.strip()):
    from tqdm import tqdm
    with open(file_input) as ifile:
        for i in tqdm(ifile, desc='file_input'):
            with open(file_output, 'a') as ofile: # append
                ofile.writelines('%s\n' % func(i))
```
