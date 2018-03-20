```sh
cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>" | sed 's\<content>\\' | sed 's\</content>\\' > news_sohusite.txt 
```
