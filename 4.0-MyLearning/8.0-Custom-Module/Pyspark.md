# Pyspark

---
- 依赖包
```
## -*- coding: utf-8 -*-
import re
import time
import numpy as np
import pandas as pd
from scipy import stats

from pyspark.sql import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
```

---
- 初始化
```
spark = SparkSession.builder \
    .appName("Yuanjie_Test") \
    .config('log4j.rootCategory',"WARN") \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext
```

---
- 常用小函数
```
# mode
Mode = udf(lambda x:float(stats.mode(x)[0]),FloatType())

# 组内众数（sql方式）
def spark_mode(df,colnames):
    gr_count = pp.groupBy(colnames).count()
    mode = gr_count.join(gr_count.groupBy(colnames[0]).agg(F.max(F.col('count')).alias('count')),on=[colnames[0],'count'])#.drop('count')
    mode.show(5)
    return mode

# median
Median = udf(lambda x:float(np.median(x)),FloatType())

# Shape
def Shape(df):
    shape = (df.count(),len(df.columns))
    print shape
    return(shape)

# 加索引
def addIndex(df):
    w = F.row_number().over(Window.orderBy(F.lit(0)))
    return(df.withColumn('_index', w))

# Cbind
def Cbind(df1,df2,how='inner',_index=False):
    """
    One of `inner`, `outer`, `left_outer`, `right_outer`, `leftsemi`
    """
    if _index:
        df = addIndex(df1).join(addIndex(df2),on='_index',how=how)
    else:
        df = addIndex(df1).join(addIndex(df2),on='_index',how=how).drop('_index')
    return(df)
```

---
- 预处理函数
```
# 特征名
def Feature_names(df,feature_string=[],feature_number=[],other=['acct_no','label']):
    if feature_string:
        feature_number = list(set(df.columns)- set(feature_string+other))
        return([feature_string,feature_number])
    if feature_number:
        feature_string = list(set(df.columns)- set(feature_number+other))
        return([feature_string,feature_number])

# 缺失值
def Imputer(df,feature_string=[],feature_number=[]):
    for i in [i[0] for i in df.select(feature_string).dtypes if i[1] !='string']:
        df = df.withColumn(i,df[i].astype(LongType()))
    for i in feature_string:
        df = df.withColumn(i,df[i].astype(StringType()))  
    for i in feature_number:
        df = df.withColumn(i,df[i].astype(FloatType()))
    df = df.fillna(dict(zip(feature_string,['88888888']*len(feature_string)) + zip(feature_number,[0]*len(feature_number))))
    return(df)

# OneHot
## label为数值型，id为字符型
def Preprocessing(df,_id='acct_no',feature_string=[],feature_number=[],model_Data='fbidm.df',isTrainData=True):
    indexed_Data = model_Data+'_indexed_Data'
    scaled_Data = model_Data+'_scaled_Data'
    df = df.withColumnRenamed(_id,'_id').withColumn('_id',col('_id').astype(StringType()))
    if 'label' in df.columns:
        df = df.withColumn('label',col('label').astype(IntegerType()))
    for i in [i[0] for i in df.select(feature_string).dtypes if i[1] !='string']:
        df = df.withColumn(i,df[i].astype(LongType()))    
    for i in feature_number:
        df = df.withColumn(i,col(i).astype(FloatType()))
    for i in feature_string:
        df = df.withColumn(i,col(i).astype(StringType()))
    print "Sucessful Data Astype:===================>10%"
    
    df = df.fillna(dict(zip(feature_string,['88888888']*len(feature_string)) + zip(feature_number,[0]*len(feature_number))))
    print "Sucessful Data Imputer:==================>20%"
    
    if feature_string:
        if isTrainData:
            DataFrameWriter(df.select(feature_string)).saveAsTable(indexed_Data,mode='overwrite')
            indexed_Data = spark.table(indexed_Data)
        else:
            indexed_Data = spark.table(indexed_Data)
            """
            在新数据集df上每类加上类别值，避免新数据各类别值变少
            """
            nrow = np.max([len(indexed_Data.select(i).rdd.countByKey()) for i in feature_string])
#           nrow = np.max([indexed_Data.select(i).drop_duplicates().count() for i in feature_string])
            df0 = spark.range(1,nrow)
            for i in feature_string:
                df0 = Cbind(df0,indexed_Data.select(i).distinct(),how='outer')
            df0 = df0.drop('id')
            df0 = df0.fillna(dict(zip(feature_string,list(df0.first()))))
            for i in df.drop(*(feature_string)).columns:
                df0 = df0.withColumn(i,lit(0))
            df0 = df0.withColumn('_id',lit('new_id')).select(df.columns)
            df = df.union(df0)
            print "Sucessful Data kinds:====================>30%"

    print "Sucessful Data Indexed:==================>40%"

    for i in feature_string:
        inputCol=i
        outputCol='indexed_'+i
        indexer = StringIndexer(inputCol=inputCol,outputCol=outputCol,handleInvalid='skip').fit(indexed_Data)###model
        df = indexer.transform(df).withColumn(i,col(outputCol)).drop(outputCol)
    print "Sucessful Data Indexing:=================>50%" 

    for i in feature_string:
        inputCol=i
        outputCol='encoded_'+i
        encoder =  OneHotEncoder(dropLast=False,inputCol=inputCol,outputCol=outputCol)
        df = encoder.transform(df).withColumn(i,col(outputCol)).drop(outputCol)
    print "Sucessful Data OneHot:===================>60%"

    vecAssembler = VectorAssembler(inputCols=feature_string+feature_number,outputCol='features')
    df = vecAssembler.transform(df)
    print "Sucessful Data VectorAssembler:==========>70%"

    if 'label' in df.columns:
        df = df.select('_id','label','features')
    else:
        df = df.select('_id','features')
    if isTrainData:
        DataFrameWriter(df).saveAsTable(scaled_Data,mode='overwrite')
        scaled_Data = spark.table(scaled_Data)
    else:
        df = df.filter(col('_id') != 'new_id')
        scaled_Data = spark.table(scaled_Data)
    print "Sucessful Data Scaled:===================>80%"

    standardScaler = StandardScaler(withMean=True, withStd=True,inputCol='features', outputCol='scaled').fit(scaled_Data)###model
    df = standardScaler.transform(df).withColumn('features',col('scaled')).drop('scaled')
    print "Sucessful Data Scaling:==================>99%"
    return(df)
```

---
```
if __name__ == '__main__':
    print('I IS MAIN!!!')
else:
    print('Sucessful Import Me!!!')
```