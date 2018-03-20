## Install
- [cmake][1]
```sh
tar zxvf cmake-3.6.0-Linux-x86 64.tar.gz 
export PATH=$PATH:/home/bnu/cmake-3.6.0-Linux-x86 64/bin
```
- [opencc][2]：[下载rpm][3]
```sh
tar -xzvf opencc-1.0.4.tar.gz
cd opencc-1.0.4/
make
sudo make install
```


```sh
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python setup.py install

WikiExtractor.py -b 2G --processes 30 -o ./zhwiki ./zhwiki-latest-pages-articles.xml.bz2
opencc -i wiki_00 -o zh_wiki_00 -c zht2zhs.ini
```

---
[1]: https://cmake.org/download/
[2]: https://bintray.com/package/files/byvoid/opencc/OpenCC
[3]: https://mirrors.aliyun.com/centos/7.4.1708/os/x86_64/Packages/opencc-tools-0.4.3-3.el7.x86_64.

