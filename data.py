import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# csv_data = '''A,B,C,D
# 1,2,3,4
# 5,6,,8
# 0,11,12,'''


# df = pd.read_csv(StringIO(csv_data))
# print(df)
# #统计为空的数目
# print(df.isnull().sum())
# print(df.values)
# #丢弃空的
# print(df.dropna())
# print('after', df)




# # 使用sklearn对Missing value进行差值计算填补
# imr = Imputer(missing_values=np.nan, strategy='mean')
# imr.fit(df) # fit 构建得到数据
# imputed_data = imr.transform(df.values) #transform 将数据进行填充
# print(imputed_data)





df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class3'],
                   ['blue', 'XL', 15.3, 'class1'],
                   ['green', 'L', 12.9, 'class2']])
df.columns =['color', 'size', 'price', 'classlabel']
print(df)


# #####################
# #类别变量转化为数值变量
# #####################

# # 方法一 mapping 
# # 1) 对特征项size定义映射字典
# size_mapping = {"M":1, "L":2, "XL":3}
# # 使用DataFrame类中的map方法进行数值转换
# df["size"]=df["size"].map(size_mapping)
# # 2) 通过枚举的方式来构建classlabel项的映射字典
# ## 遍历Series
# print(np.unique(df["classlabel"]))
# for idx, label in enumerate(df['classlabel']):
#   print(idx, label)

# classlabel_mapping = {label:idx for idx, label in enumerate(np.unique(df["classlabel"]))}
# df["classlabel"]=df["classlabel"].map(classlabel_mapping)
# print(df)



# # 方法二 使用sklearn中LabelEncoder()类进行映射
# class_le = LabelEncoder()
# class_le.fit(df["classlabel"])
# print(class_le.classes_)
# df["classlabel"]= class_le.fit_transform(df["classlabel"])
# print(df)
# print(class_le.inverse_transform(df["classlabel"]))


# 方法三 使用sklearn中的OrdinalEncoder来进行编码
size_oe = OrdinalEncoder()
size_oe.fit(df['size'].values.reshape(-1,1))
size_new = size_oe.fit_transform(df['size'].values.reshape(-1,1))
print(size_oe.categories_)
print(size_new)
print(size_oe.inverse_transform(size_new))


# # 方法四 One-hot方法
# # 为什么要使用One-hot方法？
# # 对于color特征项，如果使用上述两种方法转换为数值型，会引入由不同的数值大小造成的特征不平等问题
# # 1) 使用pandas中的get_dummies()方法（哑变量）来进行处理
# pf = pd.get_dummies(df["color"])
# df = pd.concat([df,pf], axis=1)
# df.drop(["color"], axis = 1,inplace = True)
# print(df)

# 2) 使用sklearn中的OneHotEncoder来进行处理
color_ohe = OneHotEncoder(sparse=False) # sparse = False意味着输出的是numpy.ndarray, True输出的是scipy.sparse.csr.csr_matrix
color_ohe.fit(df["color"].values.reshape(-1,1))
color_New = color_ohe.fit_transform(df["color"].values.reshape(-1,1))
col_name = ["color_"+s for s in color_ohe.categories_[0]]                   #color_ohe.categories_是一个list类型变量，其中第一项是一个包含了列名的numpy数组，这里这样做是为了后面生成的pd DataFrame的列名没有括号
print(col_name)
dfColor_ohe = pd.DataFrame(color_New, columns= col_name)

print(dfColor_ohe.columns)
df = pd.concat([df,dfColor_ohe], axis=1)
df.drop(["color"], axis=1, inplace = True)
print(df)


############
# References
############

# http://www.insightsbot.com/blog/McTKK/python-one-hot-encoding-with-scikit-learn
# https://blog.csdn.net/HHTNAN/article/details/80237769