[TOC]



------





# 数据预处理

## 类型转换
```python
from typing import List
def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df

def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))
```

## 字符串处理

### 字符串截取

```python
'''
df.loc[df['col1'].str.contains('str1'),'col2']='str2'
'''

```


## 时间处理

### 两个日期的月份差（计算车月龄）

```python
def months(str1,str2):
    import datetime
    year1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").year
    year2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").year
    month1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d").month
    month2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d").month
    num=(year1-year2)*12+(month1-month2)
    return num
```

### 时区加减

```python
def time_change(datin,num): #加减固定数值（天，时分秒）
    '''
    num:需要加减的天数、时、分、秒
    '''
    import datetime
    for each in datin.columns:
        if datin[each].dtypes=='datetime64[ns]':
            datin1[each]=datin[each].dropna().apply(lambda x: (datetime.datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S")+\
                                                datetime.timedelta(hours=num)).strftime("%Y-%m-%d %H:%M:%S"))
    return datin1
#hours 加减号可改       
```

### 截取时间

```python
def time_intercept(datin,date_list,date_format): # "%Y-%m-%d %H:%M:%S"
    '''
    time_intercept(datin,['col1','col2',...],'%d') 
    date_list:需要截取时间的列名
    date_format:截取后的时间格式，"%Y-%m-%d %H:%M:%S"之间的都可取
    '''
    import datetime
    for each in date_list:
        datin[each]=datin[each].astype('datetime64[ns]') #格式必须为datetime64
        datin['{}_{}'.format(each,date_format)]=datin[each].dropna().apply(lambda x: x.strftime('{}'.format(date_format))) #要去除空值
    return datin
```

### 日期转星期

```python
#转换为星期
def date_to_week(datin,date_col):
    '''
    date_col:需要转为星期的列名
    '''
    import datetime
    datin['{}_week'.format(date_col)] = datin[date_col].dropna().apply(lambda x: datetime.datetime.strptime(x.strftime('%Y-%m-%d'),"%Y-%m-%d").weekday()+1)
    return datin
```

### 减去某一天求时间差

```python
def diff_days(datin,date_col,cut_date):
    '''
    date_col:需要转为星期的列名
    cut_date:eg.'2019-09-13'
    '''
    datin[date_col]=datin[date_col].astype('datetime64[ns]')
    datin[date_col]=datin[date_col].dropna().apply(lambda x: x.strftime("%Y-%m-%d"))

    cut_date1=datetime.datetime.strptime(cut_date,'%Y-%m-%d')


    datin['diff_days']=datin[date_col].dropna().\
    apply(lambda x: (datetime.datetime.strptime(x,'%Y-%m-%d')-cut_date1).days)
    return datin
#datin1=diff_days(datin,'到店时间','2019-09-13')
```

## 编码

### one_hot(sklearn)

```python 
from typing import List
def one_hot_encoder(datin:pd.DataFrame,cat_features: List[str] = []):
    '''
    cat_features:需要编码的特征名list
    '''
    from sklearn.preprocessing import OneHotEncoder
    for each in cat_features:
        datin[each]=datin[each].fillna('-1') #默认用-1填充缺失

    encoded_features = []
    dfs=[datin]
    for df in dfs:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
            n = df[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)
    datin=pd.concat([datin, *encoded_features[:len(cat_features)]], axis=1) #len(),选择拼接几个特征的one_hot编码
    return datin

#cat_features = ['特约店所属大区','性别']
#one_hot_encoder(datin,cat_features)
```

### label(sklearn)

```python 
#feature_list=['车系']
def label_encoding(datin,feature_list):
    '''
    feature_list:需要编码的特征名list
    '''
    from sklearn.preprocessing import LabelEncoder
    non_numeric_features = feature_list
    dfs=[datin]
    for df in dfs:   
        for feature in non_numeric_features:        
            df['{}_label'.format(feature)] = LabelEncoder().fit_transform(df[feature])
    return df
#datin1=label_encoding(datin,feature_list)
```
### one_hot(pandas)
```python
def dummy_encode(df,flist):
    '''
    flist:需dummy列表，列名或者索引
    返回带dummy的数据框
    --------eg-----------------------------
    flist=['store_area','gear_box'] or flist=[3,6]
    dummy_encode(datin,flist)
    '''
    if all(isinstance(x,str) for x in flist):
        dummy_df=pd.get_dummies(df[flist],prefix=flist)
        result=pd.concat([df,dummy_df],axis=1)
    if all(isinstance(x,int) for x in flist):
        dummy_df=pd.get_dummies(df.iloc[:,flist],prefix=df.iloc[:,flist].columns)
        result=pd.concat([df,dummy_df],axis=1)
    return result
```

### label(pandas)
```python
def label_encode(df,col_name):
    '''
    col_name:1-D str
    返回带label的数据框
    -----------eg-------------------
    label_encode(datin,'store_city_level')
    '''
    labels, uniques = pd.factorize(df[col_name],sort=True)
    label_encode=pd.DataFrame({'{}_labels'.format(col_name):labels})
    result=pd.concat([df.reset_index(),label_encode],axis=1)
    return result
```

### Target Encoding



### X Encoding



## 特征处理
### SumZeros
```python
def add_SumZeros(train, test, features,exclude_list):
    '''
    统计每一行0的个数作为新特征
    X_train, X_test = add_SumZeros(X_train, X_test, ['SumZeros'],exclude_list)
    '''
    flist = [x for x in train.columns if not x in exclude_list]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in exclude_list]

    return train, test
```
### SumValues
```python
def add_SumValues(train, test, features,exclude_list):
    '''
    每一行求和作为新特征
    X_train, X_test = add_SumValues(X_train, X_test, ['SumValues'],exclude_list)
    '''
    flist = [x for x in train.columns if not x in exclude_list]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in exclude_list]

    return train, test
```
### 多列统计量特征
```python
def add_OtherAgg(train, test, features,exclude_list):
    '''
    对多个列计算行统计量
    X_train, X_test = add_OtherAgg(X_train, X_test, ['OtherAgg'],exclude_list)
    '''
    flist = [x for x in train.columns if not x in exclude_list]
    if 'OtherAgg' in features:
        train['Mean']   = train[flist].mean(axis=1)
        train['Median'] = train[flist].median(axis=1)
        train['Mode']   = train[flist].mode(axis=1)
        train['Max']    = train[flist].max(axis=1)
        train['Var']    = train[flist].var(axis=1)
        train['Std']    = train[flist].std(axis=1)
        
        test['Mean']   = test[flist].mean(axis=1)
        test['Median'] = test[flist].median(axis=1)
        test['Mode']   = test[flist].mode(axis=1)
        test['Max']    = test[flist].max(axis=1)
        test['Var']    = test[flist].var(axis=1)
        test['Std']    = test[flist].std(axis=1)
    flist = [x for x in train.columns if not x in exclude_list]

    return train, test
```

### 分组每列统计量特征
```python
def group_agg(df,key,idx,fun):
    '''
    df:原数据框
    key:聚合(分组)依据
    idx:聚合后需计算统计量的列索引或列名列表
    fun:统计量列表
    ---------------eg---------------------------------
    key=['store_area','store_city_level']
    fun=['count','mean','std']
    idx=['last_kilometres', 'last_to_store_time']
    group_agg(datin,key,idx,fun)
    '''
    grouped=df.groupby(key)
    if all(isinstance(x, int) for x in idx):
        group_result=grouped[df.iloc[:,idx].columns].agg(fun)
    if all(isinstance(x, str) for x in idx):
        group_result=grouped[idx].agg(fun)
        
    group_result.columns=list(group_result.columns)
    group_result=group_result.reset_index()
    result = pd.merge(df,group_result,left_on =key,right_on = key)
    return result
```


### 多列倒数变换
```python
def inverse_convert(df,idx):
    '''
    idx:需要变换的列索引
    '''
    tmp=1/df.iloc[:,idx]
    tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],'inv') for i in range(0,len(idx))} ,inplace=True)    
    result=pd.concat([df,tmp],axis=1)             
    return result    
```

### 多列多项式变换
```python
#生成给定系数的一元多项式组合,对某些列进行多项式变换
def ployld_convert(idx,df,coff):
    '''
    idx:需要变换的列索引
    coff:给定的多项式系数（高次幂到常数项）
    '''
    c = np.array(coff) 
    p = np.poly1d(c)
    tmp=df.iloc[:,idx].apply(lambda x: p(x))#aixs=1 每行应用函数
    
    tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],'poly1d') for i in range(0,len(idx))} ,inplace=True)    
    result=pd.concat([df,tmp],axis=1)
               
    return result
```

### 多列对数变换
```python
def log_convert(df,idx,method):
    '''
    idx:需要变换的列索引
    method: 'log','log2','log10'
    '''
    if 'log' in method:
        tmp=np.log(df.iloc[:,idx])
    if 'log2' in method:
        tmp=np.log(df.iloc[:,idx])
    if 'log10' in method:
        tmp=np.log(df.iloc[:,idx])
        
    tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],method) for i in range(0,len(idx))},inplace=True)    
    result=pd.concat([df,tmp],axis=1)             
    return result 
    
```

### cut 自定义分箱统计

```python
def bins_stat(df,bins,key,value,fun):
    '''
    bins:自定义分箱点
    key:被分箱的列
    value:统计列
    fun:agg的统计函数或自定义,='count'时，与.value_counts(bins=[])功能相近
    '''
    df['{}_cut'.format(key)]=pd.cut(df[key], bins)
    return df.groupby('{}_cut'.format(key))[value].agg(fun)
```

### 多列等频/等距/自定义分箱
```python
def bin_cut(df,idx,method,**para_dict):
    '''
    idx:需要分箱的列索引
    method:'q'等频，'n'等距 'cut'自定义分箱
    **para_dict:{n:箱数 bins:自定义分箱点}
    ---------eg----------------------------
    df=datin
    idx=[10,11,12]
    method='cut'
    para_dict={'n':5,'bins':[10,50]}
    bin_cut(df,idx,method,**para_dict)
    '''
    para_dict0={'n':10,'bins':[]}
    for i in para_dict:
        if i in para_dict0:
            para_dict0[i]=para_dict[i]
    
    if 'q' in method:
        tmp=df.iloc[:,idx].apply(lambda x: pd.qcut(x,para_dict0['n'],duplicates='drop'))
        tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],'qbucket') for i in range(0,len(idx))},inplace=True)    
        result=pd.concat([df,tmp],axis=1)
        
    if 'n' in method:
        tmp=df.iloc[:,idx].apply(lambda x: pd.cut(x,para_dict0['n']))
        tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],'nbucket') for i in range(0,len(idx))},inplace=True)    
        result=pd.concat([df,tmp],axis=1)
    
    if 'cut' in method:
        tmp=df.iloc[:,idx].apply(lambda x: pd.cut(x,para_dict0['bins']))
        tmp.rename(columns={tmp.columns[i]:'{}_{}'.format(tmp.columns[i],'bins') for i in range(0,len(idx))},inplace=True)    
        result=pd.concat([df,tmp],axis=1)
        
    return result
         
```
### 等频分箱的WOE、IV
```python
#等频分箱
def bin_frequency(x,y,n=10): # x为待分箱的变量，y为target变量.n为分箱数量
    total = y.count()  # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = y.count()-y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x,n,duplicates='drop')})  # 用pd.cut实现等频分箱
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin']) 
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    cut = []
    cut.append(float('-inf'))
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe
```
### 等距分箱的WOE、IV
```python
def bin_distince(x,y,n=10): # x为待分箱的变量，y为target变量.n为分箱数量
    total = y.count()  # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = y.count()-y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x,n)}) #利用pd.cut实现等距分箱
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin']) 
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    cut = []
    cut.append(float('-inf'))
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4,iv,cut,woe
```
### 自定义分箱的WOE、IV
```python
def bin_self(x,y,cut): # x为待分箱的变量，y为target变量,cut为自定义的分箱(list)
    total = y.count()  # 计算总样本数
    bad = y.sum()      # 计算坏样本数
    good = y.count()-y.sum()  # 计算好样本数
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.cut(x,cut)}) 
    d2 = d1.groupby('bucket',as_index=True)  # 按照分箱结果进行分组聚合
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin']) 
    d3['min_bin'] = d2.x.min()  # 箱体的左边界
    d3['max_bin'] = d2.x.max()  # 箱体的右边界
    d3['bad'] = d2.y.sum()  # 每个箱体中坏样本的数量
    d3['total'] = d2.y.count() # 每个箱体的总样本数
    d3['bad_rate'] = d3['bad']/d3['total']  # 每个箱体中坏样本所占总样本数的比例
    d3['badattr'] = d3['bad']/bad   # 每个箱体中坏样本所占坏样本总数的比例
    d3['goodattr'] = (d3['total'] - d3['bad'])/good  # 每个箱体中好样本所占好样本总数的比例
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])  # 计算每个箱体的woe值
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()  # 计算变量的iv值
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True) # 对箱体从大到小进行排序
    print('分箱结果：')
    print(d4)
    print('IV值为：')
    print(iv)
    woe = list(d4['woe'].round(3))
    return d4,iv,woe
```





### k-Means
```python
def add_kMeans_f(X_train,X_test,exclude_list,n_range):
    '''
    n_range=range(2,11) 分群个数，特征值为对应群号
    exclude_list=['ID','target']
    add_kMeans_f(X_train,X_test,exclude_list,n_range)
    '''
    flist = [x for x in X_train.columns if not x in exclude_list]
    flist_kmeans = []
    for ncl in n_range:
        cls = KMeans(n_clusters=ncl)
        cls.fit_predict(X_train[flist].values)
        X_train['kmeans_cluster_'+str(ncl)] = cls.predict(X_train[flist].values)
        X_test['kmeans_cluster_'+str(ncl)] = cls.predict(X_test[flist].values)
        flist_kmeans.append('kmeans_cluster_'+str(ncl))
    print(flist_kmeans)
```
### PCA
```python
def add_pca_f(X_train,X_test,exlist,n):
    '''
    exlist=['ID','target']
    n=20 主成分数
    add_pca_f(X_train,X_test,exlist,n)
    '''
    flist = [x for x in X_train.columns if not x in exlist]
    n_components = n
    flist_pca = []
    pca = PCA(n_components=n_components)
    x_train_projected = pca.fit_transform(normalize(X_train[flist], axis=0))
    x_test_projected = pca.transform(normalize(X_test[flist], axis=0))
    for npca in range(0, n_components):
        X_train.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
        X_test.insert(1, 'PCA_'+str(npca+1), x_test_projected[:, npca])
        flist_pca.append('PCA_'+str(npca+1))
    print(flist_pca)

```

### WOE_IV(Weight of Evidence 证据权重)
```python
def compute_WOE_IV(df,col,target):
    """
    对已分箱的特征计算WOE、IV
    df:DataFrame|包含feature和label
    col:str|feature名称，col这列已经经过分箱
    taget:str|label名称,0,1
    return 每箱的WOE(字典类型）和总的IV之和,注意考虑计算时候分子分母为零的溢出情况
    compute_WOE_IV(df,'last_kilometres_qbucket','is_by_to_store')
    """
    import numpy as np
    
    total = df.groupby([col])[target].count() #计算col每个分组中的样本总数
    total = pd.DataFrame({'total': total})
    
    bad   = df.groupby([col])[target].sum()   #计算col每个分组中的目标取值为1的总数，关注的正样本
    bad   = pd.DataFrame({'bad': bad})
    
    regroup = total.merge(bad,left_index=True,right_index=True,how='left')
    regroup.reset_index(level=0,inplace=True)
    
    N = sum(regroup['total'])  #样本总数
    B = sum(regroup['bad'])    #正样本总数
    
    regroup['good'] = regroup['total'] - regroup['bad'] #计算col每个分组中的目标取值为0的总数，关注的负样本
    G = N - B #负样本总数
    
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    
    regroup["WOE"] = regroup.apply(lambda x:np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    
    WOE_dict = regroup[[col,"WOE"]].set_index(col).to_dict(orient="index")
    IV = regroup.apply(lambda x:(x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    
    IV = sum(IV)
    
    return {"WOE":WOE_dict,"IV":IV}
```

### 类别变量特征值计数
```python
def count_encode(X, categorical_features, normalize=False):
    '''
    返回新特征：类别特征各取值的计数
    X: DataFrame
    categorical_features: list str
    normalize=True: 特征各取值计数/max(特征各取值计数) 计数也可以除以最频繁的类别以获得标准化值
    -----------------------eg----------------------------------
    count_encode(datin,['gear_box','store_area'],normalize=True)
    '''   
    print('Count encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        X_[cat_feature] = X[cat_feature].astype(
            'object').map(X[cat_feature].value_counts())
        if normalize:
            X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
    X_ = X_.add_suffix('_count_encoded')
    if normalize:
        X_ = X_.astype(np.float32)
        X_ = X_.add_suffix('_normalized')
    else:
        X_ = X_.astype(np.uint32)
    #X_为count_encode结果
    result=pd.concat([X,X_],axis=1)
    return result
```
### 类别变量特征值计数排序编码
```python
def labelcount_encode(X, categorical_features, ascending=False):
    '''
    返回新特征：对类别变量特征值计数后排序编码，降低outlier的影响
    --------eg----------------------------------------------------------
    labelcount_encode(datin, ['gear_box','store_area'], ascending=False)
    '''
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_
```

### 类别变量Target Encode
```python
def target_encode(X, X_valid, categorical_features, X_test=None,
                  target_feature='target'):
    print('Target Encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    X_valid_ = pd.DataFrame()
    if X_test is not None:
        X_test_ = pd.DataFrame()
    for cat_feature in categorical_features:
        group_target_mean = X.groupby([cat_feature])[target_feature].mean()
        X_[cat_feature] = X[cat_feature].map(group_target_mean)
        X_valid_[cat_feature] = X_valid[cat_feature].map(group_target_mean)
    X_ = X_.astype(np.float32)
    X_ = X_.add_suffix('_target_encoded')
    X_valid_ = X_valid_.astype(np.float32)
    X_valid_ = X_valid_.add_suffix('_target_encoded')
    if X_test is not None:
        X_test_[cat_feature] = X_test[cat_feature].map(group_target_mean)
        X_test_ = X_test_.astype(np.float32)
        X_test_ = X_test_.add_suffix('_target_encoded')
        return X_, X_valid_, X_test_
    return X_, X_valid_
```
### 类别变量合并

### 类别变量排序
## 抽样

### 随机抽样

```python
'''
df1=df.sample(frac=0.1,random_state=0) #抽10%
df.sample(n=10) #随机抽10个
df.sample(n=10,replace=True)#有放回抽样
'''
```

### 下抽样(简单随机)

```python
def lower_sample_data(df, feature,label,percent):
    np.random.seed(0) #固定随机种子
    '''
    feature:下抽样的依据特征
    label:样本量大的样本label
    percent:多数类别下采样的数量相对于少数类别样本数量的比例(大样本/小样本)
    '''
    data1 = df[df[feature] == label]  # 将多数类别的样本放在data1
    data0 = df[df[feature] != label]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号，size=percent倍的小样本量
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))

#model_xy_lower=lower_sample_data(model_xy,'is_by_to_store',0,2)
#model_xy_lower['is_by_to_store'].value_counts()

```
### 下抽样(imblearn)
```python
def UnderSampler(X,y,method,**para_dict):
    from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss
    '''
    -------------method----------------
    https://blog.csdn.net/qq_31813549/article/details/79964973
    ClusterCentroids: 每一个类别的样本都会用K-Means算法的中心点来进行合成, 而不是随机从原始样本进行抽取.
                      该方法要求原始数据集最好能聚类成簇. 此外, 中心点的数量应该设置好, 这样下采样的簇能很好地代表原始数据.
    RandomUnderSampler:随机选取数据的子集.
    NearMiss:启发式(heuristic)的规则来选择样本, 通过设定version参数来实现三种启发式的规则
             NearMiss-1: 选择离N个近邻的负样本的平均距离最小的正样本;
             NearMiss-2: 选择离N个负样本最远的平均距离最小的正样本;
             NearMiss-3: 是一个两段式的算法. 首先, 对于每一个负样本, 保留它们的M个近邻样本; 接着, 那些到N个近邻样本平均距离最大的正样本将被选择.
    
    ---eg-------------------
    para_dict={'random_state':5}
    X_resampled, y_resampled=UnderSampler(X,y,'EditedNearestNeighbours',**para_dict)
    sorted(Counter(y_resampled).items())
       
    --------para-------------------------------
    sampling_strategy :
    estimator : object, default=KMeans()
    Pass a :class:`sklearn.cluster.KMeans` estimator.
    voting : {"hard", "soft", "auto"}, default='auto'
    Voting strategy to generate the new samples:           
    - If ``'hard'``, the nearest-neighbors of the centroids found using the
      clustering algorithm will be used.
    - If ``'soft'``, the centroids found by the clustering algorithm will
      be used.
    - If ``'auto'``, if the input is sparse, it will default on ``'hard'``
      otherwise, ``'soft'`` will be used.
      
    'replacement': Whether the sample is with or without replacement.
    
    kind_sel : {'all', 'mode'}, default='all'
    Strategy to use in order to exclude samples.

    - If ``'all'``, all neighbours will have to agree with the samples of
      interest to not be excluded.
    - If ``'mode'``, the majority vote of the neighbours will be used in
      order to exclude a sample.
      
    '''
    para_dict0={'sampling_strategy':'auto','random_state':None,'estimator':None,'voting':'auto','n_jobs':None,
                'replacement':False,
                'version':1,'n_neighbors':3,'n_neighbors_ver3':3,
                'kind_sel':'all',
                }
    
    for i in para_dict:
        if i in para_dict0:
            para_dict0[i]=para_dict[i]  
            
    if 'ClusterCentroids' in method:
        X_resampled, y_resampled = ClusterCentroids(sampling_strategy=para_dict0['sampling_strategy'], 
                                         random_state=para_dict0['random_state'],
                                         estimator=para_dict0['estimator'],
                                         voting=para_dict0['voting'],
                                         n_jobs=para_dict0['n_jobs']
                                                   ).fit_sample(X, y)    

    if 'RandomUnderSampler' in method:
        X_resampled, y_resampled = RandomUnderSampler(sampling_strategy=para_dict0['sampling_strategy'], 
                                         random_state=para_dict0['random_state'],
                                         replacement=para_dict0['replacement']
                                                     ).fit_sample(X, y)
                                     
    if 'NearMiss' in method:
        X_resampled, y_resampled =NearMiss(sampling_strategy=para_dict0['sampling_strategy'],
                                           version=para_dict0['version'],
                                           n_neighbors=para_dict0['n_neighbors'],
                                           n_neighbors_ver3=para_dict0['n_neighbors_ver3'],
                                           n_jobs=para_dict0['n_jobs'],).fit_sample(X, y)
        
    print(para_dict0)    
    return X_resampled, y_resampled
```

### 带清洗下抽样（imblearn）
```python
def CleanUnderSampler(X,y,method,**para_dict):
    '''
    --Cleaning under-sampling techniques
    https://blog.csdn.net/qq_31813549/article/details/79964973
    -----eg-----------------
    para_dict={'random_state':5}
    X_resampled, y_resampled=CleanUnderSampler(X,y,'InstanceHardnessThreshold',**para_dict)
    sorted(Counter(y_resampled).items())
    
    '''
    from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,CondensedNearestNeighbour
    from imblearn.under_sampling import OneSidedSelection,NeighbourhoodCleaningRule,InstanceHardnessThreshold
    from sklearn.linear_model import LogisticRegression
    
    para_dict0={'sampling_strategy':'auto','random_state':None,'n_jobs':None,'n_neighbors':3,'kind_sel':'all','max_iter':100,
                'allow_minority':False,'n_seeds_S':1,'threshold_cleaning':0.5,'estimator':LogisticRegression(),'cv':5}
    
    for i in para_dict:
        if i in para_dict0:
            para_dict0[i]=para_dict[i] 
            
    if 'TomekLinks' in method:
        X_resampled, y_resampled =TomekLinks(sampling_strategy=para_dict0['sampling_strategy'], 
                                             n_jobs=para_dict0['n_jobs']
                                            ).fit_sample(X, y)
        
    if 'EditedNearestNeighbours' in method:    
        X_resampled, y_resampled =EditedNearestNeighbours(sampling_strategy=para_dict0['sampling_strategy'],
                                                          n_neighbors=para_dict0['n_neighbors'],
                                                          kind_sel=para_dict0['kind_sel'],
                                                          n_jobs=para_dict0['n_jobs'],).fit_sample(X, y)
    if 'RepeatedEditedNearestNeighbours' in method:      
        X_resampled, y_resampled =RepeatedEditedNearestNeighbours(sampling_strategy=para_dict0['sampling_strategy'],
                                                                  n_neighbors=para_dict0['n_neighbors'],
                                                                  max_iter=para_dict0['max_iter'],
                                                                  kind_sel=para_dict0['kind_sel'],
                                                                  n_jobs=para_dict0['n_jobs'],
                                                                 ).fit_sample(X, y)
    if 'AllKNN' in method:    
        X_resampled, y_resampled =AllKNN(sampling_strategy=para_dict0['sampling_strategy'],
                                         n_neighbors=para_dict0['n_neighbors'],
                                         kind_sel=para_dict0['kind_sel'],
                                         allow_minority=para_dict0['allow_minority'],
                                         n_jobs=para_dict0['n_jobs'],
                                        ).fit_sample(X, y)
        
    if 'CondensedNearestNeighbour' in method:     
        X_resampled, y_resampled =CondensedNearestNeighbour(sampling_strategy=para_dict0['sampling_strategy'],
                                                            random_state=para_dict0['random_state'],
                                                            n_neighbors=para_dict0['n_neighbors'],
                                                            n_seeds_S=para_dict0['n_seeds_S'],
                                                            n_jobs=para_dict0['n_jobs'],
                                                           ).fit_sample(X, y)
    
    if 'OneSidedSelection' in method:     
        X_resampled, y_resampled =OneSidedSelection(sampling_strategy=para_dict0['sampling_strategy'],
                                                    random_state=para_dict0['random_state'],
                                                    n_neighbors=para_dict0['n_neighbors'],
                                                    n_seeds_S=para_dict0['n_seeds_S'],
                                                    n_jobs=para_dict0['n_jobs'],
                                                   ).fit_sample(X, y)
        
    if 'NeighbourhoodCleaningRule' in method:     
        X_resampled, y_resampled =NeighbourhoodCleaningRule(sampling_strategy=para_dict0['sampling_strategy'],
                                                            n_neighbors=para_dict0['n_neighbors'],
                                                            kind_sel=para_dict0['kind_sel'],
                                                            threshold_cleaning=para_dict0['threshold_cleaning'],
                                                            n_jobs=para_dict0['n_jobs'],
                                                           ).fit_sample(X, y)
        
    if 'InstanceHardnessThreshold' in method:     
        X_resampled, y_resampled =InstanceHardnessThreshold(estimator=para_dict0['estimator'],
                                                            sampling_strategy=para_dict0['sampling_strategy'],
                                                            random_state=para_dict0['random_state'],
                                                            cv=para_dict0['cv'],
                                                            n_jobs=para_dict0['n_jobs'],
                                                           ).fit_sample(X, y)
    print(para_dict0)    
    return X_resampled, y_resampled 
```

### 分层抽样
```python
def typicalsamling(group,typicalNDict):
    '''
    typicalNDict={0:0.4,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.1}
    result=datin.groupby('store_city_level',group_keys=False).apply(typicalsamling,typicalNDict)
    '''
    name=group.name
    n=typicalNDict[name]
    return group.sample(frac=n) #frac=n:按比例抽样，n=n按个数抽样

```

### 上抽样（SMOTE,ADASYN,RandomOverSampler）
```python
def OverSampler(X,y,method,**kwarg):
    '''
    RandomOverSampler：通过简单的随机采样少数类的样本, 使得每类样本的比例为1:1:1
    SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本;
    ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
    
    method='SMOTE','ADASYN','RandomOverSampler'
    kwarg: para_dict={'sampling_strategy':'auto','random_state':None,'k_neighbors':5,'n_jobs':None}
    sampling_strategy={0:1000,1:2000}, 
    #'float':抽样后最小集合/最大集合，仅用于二分类情况
    #'minority': resample only the minority class;
    #'not minority': resample all classes but the minority class;
    #'not majority': resample all classes but the majority class;
    #'all': resample all classes;
    #'auto': equivalent to 'not majority'.
    #'dict': keys, targeted classes. values, desired number of samples for each targeted class.
    #'callable', function taking ``y`` and returns a ``dict``.
    random_state=0,
    k_neighbors=5,
    n_jobs=None
    ----eg----------
    X_resampled, y_resampled=OverSampler(X,y,'SMOTE',**para_dict)
    sorted(Counter(y_resampled).items())
    
    '''
    from collections import Counter
    from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
    
    para_dict0={'sampling_strategy':'auto','random_state':None,'k_neighbors':5,'n_jobs':None}
    for i in kwarg:
        if i in para_dict0:
            para_dict0[i]=kwarg[i]
            
    if 'SMOTE' in method:
        X_resampled, y_resampled = SMOTE(sampling_strategy=para_dict0['sampling_strategy'], 
                                         random_state=para_dict0['random_state'],
                                         k_neighbors=para_dict0['k_neighbors'],
                                      n_jobs=para_dict0['n_jobs']).fit_sample(X, y)
        
    if 'ADASYN' in method:
        X_resampled, y_resampled = ADASYN(sampling_strategy=para_dict0['sampling_strategy'], 
                                         random_state=para_dict0['random_state'],
                                         n_neighbors=para_dict0['k_neighbors'],
                                         n_jobs=para_dict0['n_jobs']).fit_sample(X, y)
    if 'RandomOverSampler' in method:
        X_resampled, y_resampled =RandomOverSampler(sampling_strategy=para_dict0['sampling_strategy'], 
                                         random_state=para_dict0['random_state']
                                                   ).fit_sample(X, y)
                                           
    print(para_dict0)    
    return X_resampled, y_resampled
```


### 上下抽样
```python
def OverUnderSampler(X,y,method,**para_dict):
    '''
    ----------eg-------------------
    para_dict={'sampling_strategy':{0:1000},'random_state':5}
    X_resampled, y_resampled=OverUnderSampler(X,y,'SMOTETomek',**para_dict)
    sorted(Counter(y_resampled).items())
    '''
    from imblearn.combine import SMOTEENN,SMOTETomek
    para_dict0={'sampling_strategy':'auto','random_state':None,'smote':None,'enn':None,'n_jobs':None,'tomek':None,}
    
    for i in para_dict:
        if i in para_dict0:
            para_dict0[i]=para_dict[i] 
            
    if 'SMOTEENN' in method:     
        X_resampled, y_resampled =SMOTEENN(sampling_strategy=para_dict0['sampling_strategy'],
                                           random_state=para_dict0['random_state'],
                                           smote=para_dict0['smote'],
                                           enn=para_dict0['enn'],
                                           n_jobs=para_dict0['n_jobs'],
                                          ).fit_sample(X, y)
        
    if 'SMOTETomek' in method:     
        X_resampled, y_resampled =SMOTETomek(sampling_strategy=para_dict0['sampling_strategy'],
                                             random_state=para_dict0['random_state'],
                                             smote=para_dict0['smote'],
                                             tomek=para_dict0['tomek'],
                                             n_jobs=para_dict0['n_jobs'],
                                            ).fit_sample(X, y)
    print(para_dict0)    
    return X_resampled, y_resampled
```


## 统计
### 计数

```python
def key_count(df,a,key):
    '''
    对不唯一的key,groupby a 计数
    '''
    temp=df.groupby(['{}'.format(a),'{}'.format(key)]).agg('count')
    temp1=temp.reset_index()
    vin_count=temp1.groupby('{}'.format(a))['{}'.format(key)].agg(['count']).sort_values(by=['count'],ascending=False)
    vin_count.reset_index(inplace=True)
    return key_count 
#等价于 df.groupby(['a'])['key'].agg(['nunique'])
```

### 分位数

```python
def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x,n) #返回百分位数
    percentile_.__name__='percentile_%s' %n #
    return percentile_ 
```

### 截取部分字符串筛选

```python
#vin_list=['LVHFC1','LVHRW1','LVHRW2','LVHTG4','LVHFR2','LVHCV4','LVHRU180','LVHRU185','LVHRU186','LVHRU187','LVHRU188','LVHRU189']
#datin.loc[(datin['vin_num'].str[0:6].isin(vin_list))|(datin['vin_num'].str[0:8].isin(vin_list)),'is_1_5T']=1
#datin['is_1_5T']=datin['is_1_5T'].fillna(0)
#datin['is_1_5T']=datin['is_1_5T'].astype('int32')
#datin['is_1_5T'].value_counts()
```

### 列联表行占比

```python
#tmp=datin.groupby(['store_area','is_lost_new'])['vin_num'].agg(['count']).unstack('is_lost_new')
#tmp
#tmp.div(tmp.sum(1),axis=0)
```

### 截取部分字段计算统计量（行）
```python
def by_col_agg(df, features,idx):
    '''
    idx:列索引
    by_col_agg(datin,'OtherAgg',idx)
    '''
    flist = [x for x in df.columns if  x in datin.iloc[:,idx].columns]
    if 'OtherAgg' in features:
        df['Mean']   = df[flist].mean(axis=1) #axis=1:每行，axis=0:每列
        df['Median'] = df[flist].median(axis=1)
        #df['Mode']   = df[flist].mode(axis=1)
        df['Max']    = df[flist].max(axis=1)
        df['Var']    = df[flist].var(axis=1)
        df['Std']    = df[flist].std(axis=1)
        df['qt95']  = df[flist].quantile(0.95,axis=1)
    return df
```

### 字段各值比例
```python
def col_value_counts(df):
    for i in range(0,len(df.columns)):
        print(df.iloc[:,i].value_counts(normalize=True))
```


## 清洗
### 剔除稀疏列
```python
def drop_sparse(train, test,exclude_list):
    flist = [x for x in train.columns if not x in exclude_list]
    for f in flist:
        if len(np.unique(train[f]))<2: #稀疏数据：大量缺失和为0的数据
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test
```
### 剔除常值列
```python
#数值型
def constant_colsToRemove(df):
    '''
    剔除df的常值列
    '''
# check and remove constant columns
    colsToRemove = []
    for col in df.select_dtypes(exclude=['object','category','datetime64[ns]']).columns:
        if df[col].std() == 0: 
            colsToRemove.append(col)
        
    # remove constant columns in the df
    df.drop(colsToRemove, axis=1, inplace=True)

    print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
    print(colsToRemove)

#全类型
def rm_cons_cols(df):
    result=df.loc[:, (df != df.iloc[0]).any()]
    print('constant cols:\n',df.loc[:, (df == df.iloc[0]).all()].head(1))
    return result
```
### 统一quantile截断

```python
def intercept_clean(datin,qt1,qt2):
    '''
    qt1:最小分位数
    qt2:最大分位数
    '''
    for each in datin.select_dtypes(exclude=['object','category']).columns: 
        if datin[each].nunique()>7: 
            datin.loc[datin[each]<datin[each].quantile(qt1),each]=datin[each].quantile(qt1)
            datin.loc[datin[each]>datin[each].quantile(qt2),each]=datin[each].quantile(qt2)
    return datin  
```

### 去掉缺失占比>n的行

```python 
def remove_miss_row(datin,miss_ratio):
    '''
    remove_miss_row(datin,miss_ratio)
    miss_ratio:[0,1]
    '''
    datin['missing_count']=datin.T.apply(lambda x: x.isnull().sum())
    datin['num_missing_count']=datin.select_dtypes(exclude=['object','category','datetime64[ns]']).T.apply(lambda x: x.isnull().sum())
    datin['cat_missing_count']=datin.select_dtypes(include=['object','category','datetime64[ns]']).T.apply(lambda x: x.isnull().sum())
    datin['miss_ratio']=datin['missing_count']/datin.shape[1]
    datin.drop(datin[(datin['miss_ratio']>miss_ratio)].index,inplace=True)
    return datin['missing_count'].value_counts(),datin['num_missing_count'].value_counts(),datin['cat_missing_count'].value_counts()
```

### 查看/清除重复

```python
#datin[datin.duplicated()] #返回所有字段均重复的行
#datin[datin.duplicated(['col1','col2',...])] #返回部分字段值都重复的记录
#datin.drop_duplicates() #清除完全重复的记录
#datin.drop_duplicates(['col1','col2',...]) #清除部分字段重复的记录
```

## 填充
### 各字段缺失统计

```python
def check_missing_data(df):
    return df.isnull().sum().sort_values(ascending=False)
```
### 缺失值替换成np.nan
```python
def replace_with_na(data_name, column_name, missval):
    return data_name[column_name].replace(missval, np.nan, inplace= True)
```

### category填充by mode/fill_value
```python
def cat_fill(datin,cat_col,fill_value):
    '''
    cat_col:需要填充的category列名
    fill_value:填充值（string）
    '''
    datin[cat_col]=datin[cat_col].cat.add_categories(fill_value)
    datin[cat_col]=datin[cat_col].fillna(fill_value)
    return datin
```

### SimpleImpute
```python
def impute_simple(base, datain, miss_value =np.nan, filltype = 'mean'):
    '''
    简单按base填充数据
    filltype='mean','median', 'most_frequent','constant'
    '''
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=miss_value, strategy=filltype)
    imp.fit(base)
    result = imp.transform(datain)                  
    return result

```
### By-group inpute
```python
def impute_by_group(datain, bygroup, var):
    '''
    impute_by_group(df1, bygroup = ['gender','age_cohort'], var='weight') #按性别年龄填充体重
    '''
    if len(list([var]))!=1:
        raise ValueError("fill the missing one column each time")
    imputed = datain.groupby(bygroup)[var].transform(
                                                lambda grp: grp.fillna(np.mean(grp))
                                            )
    return imputed
```

### KNN impute
```python
#scikit-learn 0.22

```

### Iterative Impute
```python
#scikit-learn 0.22
def impute_iterative(base, datain):
    '''
    用其他特征填充缺失特征
    A strategy for imputing missing values by modeling each feature with missing values 
    as a function of other features in a round-robin fashion.
    base=[[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
    datain=[[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    impute_iterative(base,datain)
    '''
    # explicitly require this experimental feature
    from sklearn.experimental import enable_iterative_imputer  # noqa
    # now you can import normally from sklearn.impute
    from sklearn.impute import IterativeImputer
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(base)
    result=imp_mean.transform(datain)               
    return result
```

### From R pacakge mice

## 异常值识别
### 时序异常检测图
```python
def detect_outlier(datin,store_id='WDYN010', win=30):
    '''
    店端到店台次异常
    '''
    tmp = datin.loc[datin.store_id==store_id].sort_values(by='date_to_store')\
                 .set_index('date_to_store')[['vin_num','upper_b']] #upper_b=mean+3*std
    timeseries = tmp['vin_num']
    upper=tmp['upper_b']
    plt.rcParams['figure.figsize'] = (12,6)
    #Determing rolling statistics
    #rolmean = pd.rolling(timeseries, window=12)
    rolmean=timeseries.rolling(window=win).mean()
    #rolstd = pd.rolling_std(timeseries, window=12)
    rolstd=timeseries.rolling(window=win).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',alpha =0.5, label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    # std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    upper_b = plt.plot(upper,linestyle='--', label='Upper bound')
    plt.legend(loc='best')
    plt.title('到店台次: {}'.format(store_id))
    plt.show(block=False)
```
## 列处理
### 左右前后填充(groupby)

## 行处理
### 分组移除标记行
```python
def mask_row(x):
    '''
    标记头两行(若从大到小排序，则标记最大的两个值)
    返回0，1数组
    '''
    result = np.ones_like(x) #生成shape与x相同，值为1的数组
    result[0:2] = 0 #数组前两行值赋为0
    return result 
def rm_mask_row(df,key,value,mask_row):
    '''
    分组移除标记行（如移除每组最大的两行）
    返回移除后的数据框
    -----------eg-------------------
    rm_mask_row(df,'store_id','vin_num',mask_row)
    '''
    mask=df.groupby([key])[value].transform(mask_first).astype(bool)#0,1转换为bool类型（True,False）
    result=df.loc[mask]
    return result
```



# 模型
## 二分类评价

### 默认

```python
def classifier_model(model,X_train,X_validation,y_train,y_validation):
    #model.fit(X_train, y_train)
    prediction = model.predict(X_validation)
    proba=model.predict_proba(X_validation)[:,1]
    
    auc =  metrics.roc_auc_score(y_validation,prediction)
    auc_proba =  metrics.roc_auc_score(y_validation,proba)
    acc = metrics.accuracy_score(y_validation,prediction)
    recall = metrics.recall_score(y_validation,prediction)
    f1 =    metrics.f1_score(y_validation,prediction)
    precision = metrics.precision_score(y_validation,prediction)
    confusion_matrix=metrics.confusion_matrix(y_validation,prediction)
    tp=confusion_matrix[1][1] # tp
    fp=confusion_matrix[0][1] # fp
    fn=confusion_matrix[1][0] # fn
    tn=confusion_matrix[0][0] # tn
    
    precision1=tp/(tp+fp)#查准率 实际且预测为到店/预测为到店:
    tpr=tp/(tp+fn)#recall 正类预测正确  查全率 实际且预测为到店/实际到店
    fpr=fp/(fp+tn) # 正类预测错误 坏瓜判错/真坏瓜
    tnr=tn/(fp+tn)# 反类预测正确  坏瓜判对/真坏瓜
    fnr=fn/(tp+fn) # 反类预测错误 好瓜判错/真好瓜
    BalancedErrorRate=(fpr+fnr)/2
    print('查准率 实际且预测为到店/预测为到店:',precision1)
    print('tpr recall 查全率 实际且预测为到店/实际到店:',tpr)
    print('fpr 未到店预测错误/实际未到店:',fpr)
    print('tnr 未到店预测正确/实际未到店:',tnr)
    print('fnr 到店预测错误/实际到店:',fnr)
    print('BalancedErrorRate:',BalancedErrorRate)
    print('AUC:',auc)
    print('AUC_proba',auc_proba)
    print('Accuracy:',acc)
    print('recall(正类预测正确):',recall)
    print('f1:',f1)
    print('precision:',precision)
    print('confusion_matrix:',confusion_matrix)
    print('tn',tn)
    print('fp',fp)
    print('fn',fn)
    print('tp',tp)
    print('fp+fn',fp+fn)
    return auc,acc,recall,f1,precision
```

### 调整(recall,precision)

```python
def classifier_model_fix(model,X_train,X_validation,y_train,y_validation,p):
    #model.fit(X_train, y_train)
    prediction = model.predict(X_validation) #pred_class
    proba=model.predict_proba(X_validation)[:,1]
    df=pd.DataFrame({'proba':proba,'prediction':prediction})
    df['fix_prediction']=df['proba'].map(lambda x:1 if x>p else 0)
    
    auc_proba =  metrics.roc_auc_score(y_validation,df['proba'])
    auc = metrics.roc_auc_score(y_validation,df['fix_prediction'])
    acc = metrics.accuracy_score(y_validation,df['fix_prediction'])
    recall = metrics.recall_score(y_validation,df['fix_prediction'])
    f1 =    metrics.f1_score(y_validation,df['fix_prediction'])
    precision = metrics.precision_score(y_validation,df['fix_prediction'])
    confusion_matrix=metrics.confusion_matrix(y_validation,df['fix_prediction'])
    tp=confusion_matrix[1][1] # tp
    fp=confusion_matrix[0][1] # fp
    fn=confusion_matrix[1][0] # fn
    tn=confusion_matrix[0][0] # tn
    
    
    precision1=tp/(tp+fp)#查准率 实际且预测为到店/预测为到店:
    tpr=tp/(tp+fn)#recall 正类预测正确  查全率 实际且预测为到店/实际到店
    fpr=fp/(fp+tn) # 正类预测错误 坏瓜判错/真坏瓜
    tnr=tn/(fp+tn)# 反类预测正确  坏瓜判对/真坏瓜
    fnr=fn/(tp+fn) # 反类预测错误 好瓜判错/真好瓜
    BalancedErrorRate=(fpr+fnr)/2
    print('查准率 实际且预测为到店/预测为到店:',precision1)
    print('tpr recall 查全率 实际且预测为到店/实际到店:',tpr)
    print('fpr 未到店预测错误/实际未到店:',fpr)
    print('tnr 未到店预测正确/实际未到店:',tnr)
    print('fnr 到店预测错误/实际到店:',fnr)
    print('BalancedErrorRate:',BalancedErrorRate)
    print('AUC:',auc)
    print('AUC_proba',auc_proba)
    print('Accuracy:',acc)
    print('recall(正类预测正确):',recall)
    print('f1:',f1)
    print('precision:',precision)
    print('confusion_matrix:',confusion_matrix)
    print('tn',tn)
    print('fp',fp)
    print('fn',fn)
    print('tp',tp)
    print('fp+fn',fp+fn)
    return auc,acc,recall,f1,precision
```

## 多分类评价

### 分类模型
```python
def multi_class_eval(model,X_test,y_test):
    '''
    多分类-离散型 模型效果常用评价指标
    '''
    preds_class = model.predict(X_test)
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(y_test,preds_class) 
    print('kappa(>0.8为几乎完全一致):',kappa) # >0.8为几乎完全一致 （-1，1）
    #海明距离
    from sklearn.metrics import hamming_loss
    ham_distance = hamming_loss(y_validation,preds_class)
    print('ham_distance(越接近0，预测结果和真实结果相同):',ham_distance) #(0,1) 越接近0，预测结果和真实结果相同
    #杰卡德相似系数
    from sklearn.metrics import jaccard_similarity_score
    jaccrd_score = jaccard_similarity_score(y_validation,preds_class)
    #normalize默认为true，这是计算的是多个类别的相似系数的平均值，normalize = false时分别计算各个类别的相似系数
    print('jaccrd_score(越接近1，预测结果和真实结果相同):',jaccrd_score) #(0,1) 越接近1，预测结果和真实结果相同
```
### 回归模型
```python
def multi_class_confusion_matrix(model,model_type,X_test,y_test):
    '''
    主要用于计算用回归模型的多分类问题的混淆矩阵
    model_type:'R' 回归 'C' 分类
    model_type='C'时等价于 metrics.confusion_matrix(y_test, model.predict(y_test))
    '''
    from itertools import chain
    if 'R' in model_type:
        tmp1=pd.DataFrame({'y_true':y_test,'y_pred':model.predict(X_test)})
    if 'C' in model_type:
        tmp1=pd.DataFrame({'y_true':y_test,'y_pred':list(chain(*model.predict(X_test)))})
        
    tmp1['y_pred1']=round(tmp1['y_pred'],0).astype('int32')
    check2=np.abs(tmp1['y_true']-tmp1['y_pred1'])
    print('预测与实际不等的占比:',(check2>0).sum()/len(check2)) #预测与实际不符的占比
    print('MAE:',(abs(tmp1['y_true']-tmp1['y_pred1'])).mean()) #MAE
    tmp1['diff']=abs(tmp1['y_true']-tmp1['y_pred1'])
    tmp1.loc[tmp1['y_true']==tmp1['y_pred1'],'is_cor']=1
    tmp1.loc[tmp1['y_true']!=tmp1['y_pred1'],'is_cor']=0
    tmp2=tmp1.groupby(['y_true','diff'])['diff'].count().unstack('diff')
    print(tmp2)
    print(tmp2.div(tmp2.sum(1),axis=0))   
```

## 曲线
### learning curve
```python
def loss_metric(m, X, y): 
    return metrics.mean_squared_error(y,m.predict(X))
loss = []
iteration =np.arange(500,6500,500)
for i in iteration:
    model=lgbm.LGBMClassifier(learning_rate=0.05, n_estimators=i, max_depth=6) #n_jobs=-1
    model.fit(X_train.drop(['car_name'],axis=1), y_train,verbose = False)
    loss.append((i, loss_metric(model, X_validation.drop(['car_name'],axis=1), y_validation), \
                 loss_metric(model, X_train.drop(['car_name'],axis=1), y_train)))
    print(i)

plt.plot([i[0] for i in loss],[i[1] for i in loss], label="Test") #lgbm树个数2500时最好
#plt.plot([i[0] for i in loss],[i[2] for i in loss], label="Train") 
plt.legend()
plt.show();
```

### pr曲线

```python
def pr_curve(model,X_validation,y_validation):
    '''
    pr_curve(model_with_early_stop,X_validation,y_validation)
    '''
    y_proba=model.predict_proba(X_validation)[:,1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_validation, y_proba)
    plt.plot(precision, recall,color='red')
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.show()

def pr_curve_px(model,X_validation,y_validation):
    '''
    pr_curve_px(model_with_early_stop,X_validation,y_validation)
    plotly较慢
    '''
    y_proba=model.predict_proba(X_validation)[:,1]
    precision, recall, thresholds = metrics.precision_recall_curve(y_validation, y_proba)
    tmp=pd.DataFrame({"recall":recall,"precision":precision})
    px.scatter(tmp,'precision','recall')
    
```

### 错误曲线(catboost)

```python
def fpr_fnr_recall_curve(model,validation_pool):
    '''
    fpr_fnr_recall_curve(model_with_early_stop,validation_pool)
    '''
    import matplotlib.pyplot as plt
    from catboost.utils import get_roc_curve
    from catboost.utils import get_fpr_curve
    from catboost.utils import get_fnr_curve

    curve = get_roc_curve(model,validation_pool)
    (fpr, tpr, thresholds) = curve

    (thresholds, fpr) = get_fpr_curve(curve=curve)
    (thresholds, fnr) = get_fnr_curve(curve=curve)
    
    plt.figure(figsize=(16, 8))
    style = {'alpha':0.5, 'lw':2}

    plt.plot(thresholds, fpr, color='blue', label='FPR', **style) # 所有反类中，有多少被预测成正类（正类预测错误） 未到店预测错误/实际未到店
    plt.plot(thresholds, fnr, color='green', label='FNR', **style) #所有正类中，有多少被预测成反类（反类预测错误） 到店预测错误/实际到店
    plt.plot(thresholds, 1-fnr, color='red', label='Recall', **style) #recall=1-fnr

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Rate', fontsize=16)
    plt.title('FPR-FNR-Recall curves', fontsize=20)
    plt.legend(loc="lower left", fontsize=16);
```
### roc曲线
```python
def plot_roc_curve(fprs, tprs):
    
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(10, 10))
    
    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))
        
    # Plotting ROC for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')
    
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plotting the mean ROC
    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)
    
    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')
    
    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)
    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title('ROC Curves of Folds', size=20, y=1.02)
    ax.legend(loc='lower right', prop={'size': 13})
    
    plt.show()
    #plot_roc_curve(fprs, tprs)
```

# plot

## 电机效率工况图

## 影响因素正负柱形图

## 时序图（工单异常）
```python
def test_stationarity1(timeseries, win=30):
    plt.rcParams['figure.figsize'] = (12,6)
    #Determing rolling statistics
    #rolmean = pd.rolling(timeseries, window=12)
    rolmean=timeseries.rolling(window=win).mean()
    rolstd=timeseries.rolling(window=win).std()   
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',alpha =0.5, label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Daily value & Rolling Mean')
    plt.show(block=False)
```
## 设定特征值顺序

## 泡泡图

# 调参

## 贝叶斯优化

## 随机搜索

## 模型自带（catboost）

# Other Tips
## value_counts()
```python
'''
pd.value_counts(values,sort=True,ascending=False,normalize=False,bins=None,dropna=True)
sort:排序
ascending=False：降序
normalize=False：计算各类别百分比
bins：分箱统计,eg.bins=2:分两个bin统计个数
dropna：是否统计缺失值
'''  
```
## 调整字段顺序
```python
#列名编号: pd.DataFrame({"col_name":df.columns}).T
def adjust_col_order(datin,idx,flag):
    '''
    挑选部分列按自定义顺序移至列前/列尾
    idx=sum([[1],list(range(4,7))],[])
    flag:head,tail
    '''
    df1=datin.iloc[:,idx]
    df2=datin.drop(datin.iloc[:,idx].columns,axis=1)
    if 'head'in flag:
        result=pd.concat([df1,df2],axis=1)
    if 'tail'in flag:
        result=pd.concat([df2,df1],axis=1)
    return result
```

## 相关系数（两两查看）
```python
def feature_corr(datin,cut_num):
    '''
    cut_num:相关系数阈值[0~1]
    '''
    datin_corr = datin.drop(list(datin.select_dtypes(include=['object','category']).columns), axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    datin_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

    datin_corr.drop(datin_corr.iloc[1::2].index, inplace=True)
    datin_corr_nd = datin_corr.drop(datin_corr[datin_corr['Correlation Coefficient'] == 1.0].index)
    corr = datin_corr_nd['Correlation Coefficient'] > cut_num
    return datin_corr_nd[corr]
#feature_corr(datin,0.9) #查看相关系数>0的指标组
```

## 内存占用
```python
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
```
## 经纬度相关计算
### 经纬度间的距离
```python
import math
def get_distances(locs_1, locs_2):
    n_rows_1 = locs_1.shape[0]
    n_rows_2 = locs_2.shape[0]
    dists = np.empty((n_rows_1, n_rows_2))
    # The loops here are inefficient
    for i in np.arange(n_rows_1):
        for j in  np.arange(n_rows_2):
            dists[i, j] = get_distance_from_lat_long(locs_1[i], locs_2[j])
    return dists
def get_distance_from_lat_long(loc_1, loc_2):

    earth_radius = 3958.75

    lat_dif = math.radians(loc_1[0] - loc_2[0])
    long_dif = math.radians(loc_1[1] - loc_2[1])
    sin_d_lat = math.sin(lat_dif / 2)
    sin_d_long = math.sin(long_dif / 2)
    step_1 = (sin_d_lat ** 2) + (sin_d_long ** 2) * math.cos(math.radians(loc_1[0])) * math.cos(math.radians(loc_2[0])) 
    step_2 = 2 * math.atan2(math.sqrt(step_1), math.sqrt(1-step_1))
    dist = step_2 * earth_radius
    return dist
```
### 4S店密度统计
```python
def spherical_dist(pos1, pos2, r=3958.75):
    '''
    -----eg-----------------------
    locations_1 = np.array([[34, -81], [32, -87], [35, -83]])
    locations_2 = np.array([[33, -84], [39, -81], [40, -88], [30, -80]])
    spherical_dist(locations_1[:, None], locations_2)
    '''
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def lng_lat_dis(df,lng_col,lat_col):
    '''
    通过经纬度，计算两地距离差
    df: col1:地名 col2:lng col3:lat
    lng_col:lng列名(str)
    lat_col:lat列名(str)
    返回两地距离差
    --------eg---------------------
    lng_lat_dis(dat1,'store_lng','store_lat')
    '''
    flist=[lng_col,lat_col]
    for col in flist:
        df[col] = df[col].astype("float")   
    locations_1 = np.array(df.loc[:,flist])
    locations_2 = np.array(df.loc[:,flist])
    dis_matrix = spherical_dist(locations_1[:, None], locations_2) #调用spherical_dist()
    dat2 = pd.DataFrame(dis_matrix, columns=df.iloc[:,0], index=df.iloc[:,0])
    dat3 = dat2.stack()
    dat3.name = 'dist'
    dat3.index.names = ['name1','name2']
    dat4 = dat3.reset_index()
    return dat4

def area_density_count(df,lng_col,lat_col,bins):
    '''
    通过经纬度，计算两地距离差和4S店数量密度统计
    df: col1:地名 col2:lng col3:lat
    lng_col:lng列名(str)
    lat_col:lat列名(str)
    bins:距离分割点[5,10,20] 5公里内的店个数，10公里内的店个数，。。。。
    返回两地距离差和距离范围内个数统计
    --------eg---------------------
    bins=[5,10,20]
    dist,density=area_density_count(dat1,'store_lng','store_lat',bins)
    '''
    flist=[lng_col,lat_col]
    for col in flist:
        df[col] = df[col].astype("float")   
    locations_1 = np.array(df.loc[:,flist])
    locations_2 = np.array(df.loc[:,flist])
    dis_matrix = spherical_dist(locations_1[:, None], locations_2) #note:需调用spherical_dist()
    dat2 = pd.DataFrame(dis_matrix, columns=df.iloc[:,0], index=df.iloc[:,0])
    dat3 = dat2.stack()
    dat3.name = 'dist'
    dat3.index.names = ['name1','name2']
    dat4 = dat3.reset_index() #两地距离差
    
    df1=pd.DataFrame()    
    for i in bins:
        df2=pd.DataFrame()
        df2['within{}'.format(i)]=dat4.loc[(dat4.dist<i)&(dat4.dist>=1)&(dat4.name1 != dat4.name2)].\
        groupby('name1')['name2'].count()
        df1=pd.concat([df1,df2],axis=1)
    df1.fillna(0,inplace=True)
    df1.reset_index(inplace=True)
    return  dat4,df1 #密度统计
```

