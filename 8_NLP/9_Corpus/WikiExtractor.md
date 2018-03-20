## Install
- [cmake][https://cmake.org/download/]
- [opencc][https://bintray.com/package/files/byvoid/opencc/OpenCC]
```sh
tar -xzvf opencc-1.0.4.tar.gz
cd opencc-1.0.4/
make
sudo make install
```
或者https://mirrors.aliyun.com/centos/7.4.1708/os/x86_64/Packages/opencc-tools-0.4.3-3.el7.x86_64.rpm



```sh
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python setup.py install

WikiExtractor.py -b 2G --processes 30 -o ./zhwiki ./zhwiki-latest-pages-articles.xml.bz2
opencc -i wiki_00 -o zh_wiki_00 -c zht2zhs.ini
```

