```
def full2half(s):
    _char = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        _char.append(chr(num))
    return ''.join(_char)
```

```python
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.lower()):
    from tqdm import tqdm
    with open(file_input) as ifile:
        with open(file_output, 'a') as ofile:  # append
            for i in tqdm(ifile, desc='File Processing'):
                ofile.writelines(func(i))
    print('End!')
```

