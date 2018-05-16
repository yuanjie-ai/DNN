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
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.lower(), overwrite=True):
    """
    :param file_input:
    :param file_output:
    :param func: return str
    :param overwrite:
    :return:
    """
    import os
    from tqdm import tqdm
    assert isinstance(func('str'), str)

    if overwrite and os.path.isfile(file_output):
        os.remove(file_output)
    with open(file_input) as ifile:
        with open(file_output, 'a') as ofile:  # append
            for i in tqdm(ifile, desc='File Processing'):
                ofile.writelines(func(i))
    print('Before Processing:\n', os.popen('wc -l %s && head -n 5 %s' % (file_input, file_input)).read())
    print('After Processing:\n', os.popen('wc -l %s && head -n 5 %s' % (file_output, file_output)).read())
```

