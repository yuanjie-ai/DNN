```python
def file_processing(file_input, file_output, func=lambda x: x.strip()):
    with open(file_input) as ifile:
        for i in tqdm(ifile, desc='file_input'):
            with open(file_output, 'w') as ofile:
                ofile.writelines('%s\n' % func(i))
```
