# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:13:04 2018

@author: Fuxiao
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import operator

#u,sigma,vt=linalg.svd([[1,1],[7,7]])
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
#Data=loadExData()
#U,Sigma,VT=linalg.svd(Data)

#sigma=np.mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
#data=U[:, :3]*sigma*VT[:3,:]

#相似度计算方法
def ecludSim(inA, inB): #计算欧几里得相似度，并归一化到0~1之间
    return 1.0/(1+np.linalg.norm(inA-inB))

def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0#如果inA少于3个点，函数返回1.0，此时两个向量完全相关
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar=0)[0][1] #皮尔逊相关系数的取值范围从-1到+ 1，我们通过0.5+0.5*corrcoef(）这个函数计算，并且把其取值范围归一化到0到1之间。

def cosSim(inA, inB): #计算余弦相似度，并归一化到0~1之间
    num=float(inA*inB.T)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

#上面的三个相似度计算都是基于列向量表示的
#myMat=mat(loadExData())
#ecludsim=ecludSim(myMat[:,0],myMat[:,4])
#cossim=cosSim(myMat[:,0], myMat[:,4])
#pearsim=pearsSim(myMat[:,0], myMat[:,4])

#基于用户相似度的推荐引擎


def user_rating_data_extact():
    '''用户评分数据提取'''
    ratings = ['user_id', 'movie_id', 'rating', 'timestamp'] #用户表的数据字段名
    users = pd.read_table('C:\\Users\\Fuxiao\\Desktop\\待选数据集\\ml-1m\\ratings.dat', sep='::', header=None, names=ratings)    

    num_users=6041 #用户数量,由于python中下表从0开始，这里加1来适应用户ID从1开始
    num_movies=3953 #电影数量，同上
    movie_rating_temp=np.zeros((num_users,num_movies), dtype=np.int) #创建二维的电影_评分列表


    usertemp=users['user_id'] #把users这个表格里面的user_id提出来
    userid=list(usertemp) #把表格user_id数据变成list形式

    for k in range(1,1000209): #从上到下遍历整个用户id，电影id和每个用户对每个电影的评分，这里产生的数据第一行和第一列均为0
        movie_rating_temp[users['user_id'][k]][users['movie_id'][k]]=users['rating'][k]

    movie_rating_1=np.delete(movie_rating_temp, 0, axis=0) #删除掉第0行
    movie_rating=np.delete(movie_rating_1, [0],axis=1)#删除掉第0列


    dataframe = pd.DataFrame(movie_rating)

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("movie_rating.csv",index=False,sep=',') #把每个用户的评分保存下来
    ############################


#sim=pearsSim(movie_rating[1], movie_rating[2])     

def cal_sim(movie_rating, sim_method=ecludSim):
    '''计算所有用户之间的相似度'''
    #movie_rating:用户评分列表， sim_method:默认的相似度计算方法是欧氏距离相似度计算
    user_sim=np.zeros([len(movie_rating), len(movie_rating)], dtype=np.float32)
    for i in range(len(movie_rating)):
        for j in range(len(movie_rating)):
            if user_sim[i,j]!=0:
                continue
            elif i==j: user_sim[i,j]=0 #把自己跟自己的相似度置为0
            else:
                user_sim[i,j]=sim_method(movie_rating[i], movie_rating[j])
                user_sim[j,i]=user_sim[i,j]
    return user_sim #返回所有用户的之间的相似度,这里可以做成一个表格保存下来


def recommend_user(movie_rating, user_sim, user, ratio=0.90): 
    '''为给定的用户进行推荐前N个电影'''
    
    #movie_rating:用户的评分列表
    #user_所有用户之间的相似程度列表
    #user:需要推荐的用户
    #ratio：需要推荐相似用户的前N占比
    
    #这里先简单处理，给一个user的ID就行，后面给具体的名字，
    #推荐的电影也是先给一个ID,后面根据ID找到名字
    
    #先从user_sim这个相似度表格里面找到最相似的前几个用户,这个相似度用前面N个用户的占比来表示，超过这个占比即可
    
    user_sim=user_sim.tolist() #转成list形式
    
    sim_sort=copy.copy(user_sim[user])
    dic={}
    for item, values in enumerate(sim_sort):#把与uer相似的用户写成 一个字典形式，因为要排序之后的索引
        dic[item]=values
    
    sim_sort=sorted(dic.items(), key=operator.itemgetter(1),reverse=True)#对当前这个用户相似的用户进行排序,从大到小
    
    
    sim_total=sum(user_sim[user]) #对所有的相似度求和
    sim_ratio=0
    N=0
    while((sim_ratio<ratio) and (N<10)): #求前N个相似度之和大于等于ratio
       #如果用户之间都不是很相似，N会很大，所以这里当需要推荐的用户个数超过10个就停止
       sim_ratio+=sim_sort[N][1]/sim_total
       N+=1
       
    sim_top=[] #构造一个前N相似度的列表,
    #sim_top.append(movie_rating[user]) #把需要推荐的用户的评分信息放在第一个
    for j in range(N):
        #找到排序前N的用户的索引，然后在用户评分列表里面找到用户评分电影的数据添加成一个列表
        user_sim_id=sim_sort[j][0]
        sim_top.append(movie_rating[user_sim_id])
    
    ind=[]
    for i in enumerate(movie_rating[user]): #找到user评分为0的索引，即用户没有看过的电影
        if i[1]==0:
            ind.append(i[0])
            
    #两种推荐方式，第一种，从与user最相似的第一个用户开始，找这个最相似用户看过的推荐，并按这个顺序来安排推荐顺序
    recommend1=[] #构造一个推荐列表
    for m in range(N):
        for n in ind:
            if sim_top[m][n]!=0:
                recommend1.append(n) #如果user没看过但与他相似的用户看过，就添加到推荐列表里面去
    recommend_one=[]
    [recommend_one.append(i) for i in recommend1 if not i in recommend_one] #去掉重复的元素并保持顺序不变
    ##############################
    
    #第二种，根据求得的前N个用户，把他们看做一个平等的，user没看过但与他相似的用户看过，只要有就对该电影加1，然后按照电影次数最多的推荐
    recommend2={}
    for m2 in recommend1:
        if m2 not in recommend2.keys():
            recommend2[m2]=0
        recommend2[m2]+=1

        
    recommend22=sorted(recommend2.items(), key=operator.itemgetter(1))#按照电影看的次数从小到大排序
    
    recommend_two=[]
    for i in recommend22:
        recommend_two.append(i[0]) #把排序后的字典的键值取出来，这个键值就是电影的ID
    recommend_two.reverse() #从大到小排序
    ###########################
    
    return recommend_one, recommend_two #把两种推荐方案都返回




#********************  2   ******************
def user_movie_genres(movie_rating, movies):#把每个用户每一个看过的电影的流派按照电影评分值相加保留
    #movie_rating: 用户-电影评分列表
    #movies: 每部电影的类型信息
    
    #temp=np.array(movies)
    #movies=temp.tolist() #转成list形式
    
    #将电影-流派写成一个列表，查看每个用户看过的电影流派有哪些，并且以用户对电影评分的高低来记录看过的电影类型值
    movies_genres=pd.DataFrame(np.zeros([6040, 18]), index=[i for i in range(6040)], columns=['Action', 'Adventure', 'Animation','Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']) #建立电影-类型的列表
    di={}#创建一个空字典放电影编号和流派
    for i in range(len(movies['movie_id'])):
        di[movies['movie_id'][i]-1]=movies['genres'][i] #电影id是从1开始的，为了统一，里面减了一个1，从0开始
    
    for i in range(len(movie_rating)): #遍历用户评分表
        watched_id=np.nonzero(movie_rating[i])  #找到当前第i个用户看过的电影编号
        
        for j in watched_id[0]: #遍历第i个用户看过的电影集
            temp_genres=di[j].split('|') #将每个电影的电影流派取出来
            for k in temp_genres: #对每个用户遍历movies_genres电影-流派矩阵，看过这个电影的电影流派加分
                movies_genres.iloc[i][k]+=movie_rating[i][j] #把第i个用户当前看过的电影流派按照这部电影的评分值累加
    dataframe = pd.DataFrame(movies_genres)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("movies_genres.csv",index=False,sep=',') #把每个用户看过的每部电影的流派值保存下来

    temp_genres=np.array(movies_genres)
    movie_genres_sim=cal_sim(temp_genres, sim_method=ecludSim)#计算基于电影类型的用户之间的相似度
    
    
    dataframe = pd.DataFrame(movie_genres_sim)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("movie_genres_based_sim.csv",index=False,sep=',') #保存用户-特征矩阵
    
    return movie_genres_sim
        
        









#********************  3   ******************

#建立用户-特征列表
def users_charactor_list(users_ch):
    #users_ch: 用户特征，如地区，性别，年龄的列表
    
    #将用户-特征写成一个列表，分类型整理一下数据
    user_charactor=pd.DataFrame(np.zeros([6040, 30]), index=[i for i in range(6040)], columns=['F','M', '1', '18', '25','35','45','50','56','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
    for i in range(len(users_ch)):
        #user_charactor.iloc[i]['user_id']=i+1 #把用户的userid放到user_charactor里面
        user_charactor.iloc[i][users_ch.iloc[i]['Gender']]=1 #性别，满足置1
        user_charactor.iloc[i][str(users_ch.iloc[i]['Age'])]=1 #年龄，满足置1
    for i in range(len(users_ch)):
        user_charactor.iloc[i][str(users_ch.iloc[i]['Occupation'])]=1 #职业，满足置1

    dataframe = pd.DataFrame(user_charactor)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("user_charactor.csv",index=False,sep=',') #先保存用户-特征矩阵
      
    temp_charactor=np.array(user_charactor)
    user_charactor_sim=cal_sim(temp_charactor, sim_method=ecludSim)#计算基于用户的性别、年龄、职业的相似度
    
    
    dataframe = pd.DataFrame(user_charactor_sim)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("user_charactor_based_sim.csv",index=False,sep=',') #保存用户-特征矩阵
    
    return user_charactor_sim #返回基于用户的性别、年龄、职业求得的相似度




'''1.基于用户-评分的推荐'''

#user_rating_data_extact() #用户对每部电影评分数据的提取

#从文件里面读取数据的时候多加了一行，这个是数据格式决定的，加了索引，得删掉
movie_rating_t= pd.read_table('G:\\tensfolw\\机器学习（普通）\\推荐系统\\movie_rating.csv', sep=',', header=None)
temp_rating=np.array(movie_rating_t)
movie_rating=np.delete(temp_rating, 0, axis=0) #删除掉第0行

##调用计算用户相似度的函数并保存每两个用户之间的相似度
#user_sim=cal_sim(movie_rating, sim_method=ecludSim) 
#dataframe = pd.DataFrame(user_sim)
#将DataFrame存储为csv,index表示是否显示行名，default=True
#dataframe.to_csv("user_rating_based_sim.csv",index=False,sep=',') #把每个用户的相似度保存下来

#在基于用户评分求得的相似度基础上。进行相似用户的推荐
user_sim_t= pd.read_table('G:\\tensfolw\\机器学习（普通）\\推荐系统\\user_rating_based_sim.csv', sep=',', header=None) #读取用户相似度列表
temp=np.array(user_sim_t)
user_sim=np.delete(temp, 0, axis=0) #读取数据的时候自动添加了一行，这里去掉。删除掉第0行
re1_user_rating_based, re2_user_rating_based=recommend_user(movie_rating, user_sim, 4, ratio=0.80) #返回两种推荐方法都使用的列表



'''2.基于电影类型计算用户之间的相似度并推荐'''
#**************************************
#读取电影-类型列表
movies_information=['movie_id', 'title', 'genres'] #电影列表的字段名
movies=pd.read_table('C:\\Users\\Fuxiao\\Desktop\\待选数据集\\ml-1m\\movies.dat', sep='::', header=None, names=movies_information)

#***************************************
#计算相似度并进行推荐
#movies_genres_sim=user_movie_genres(movie_rating, movies) #构建用户-流派统计表
movie_genres_sim= pd.read_table('G:\\tensfolw\\机器学习（普通）\\推荐系统\\movie_genres_based_sim.csv', sep=',', header=None) #读取用户相似度列表
temp_movie_genres_sim=np.array(movie_genres_sim)
movie_genres_sim=np.delete(temp_movie_genres_sim, 0, axis=0) #读取数据的时候自动添加了一行，这里去掉。删除掉第0行
re1_movie_genres_based, re2_movie_genres_based=recommend_user(movie_rating, movie_genres_sim, 4, ratio=0.80) #返回两种推荐方法都使用的列表



'''3.基于用户的性别、年龄、职业等计算用户之间的相似度并进行推荐'''
#读取用户-特征列表
user_information=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'] #字段名
users_ch=pd.read_table('C:\\Users\\Fuxiao\\Desktop\\待选数据集\\ml-1m\\users.dat', sep='::', header=None, names=user_information)

#user_charactor_sim=users_charactor_list(users_ch)  #产生用户-特征相似度矩阵


#在基于用户的性别、年龄、职业求得的相似度基础上，进行相似的推荐

user_charactor_sim= pd.read_table('G:\\tensfolw\\机器学习（普通）\\推荐系统\\user_charactor_based_sim.csv', sep=',', header=None) #读取用户相似度列表
temp_user_charactor_sim=np.array(user_charactor_sim)
user_charactor_sim=np.delete(temp_user_charactor_sim, 0, axis=0) #读取数据的时候自动添加了一行，这里去掉。删除掉第0行
re1_user_charactor_based, re2_user_charactor_based=recommend_user(movie_rating, user_charactor_sim, 4, ratio=0.80) #返回两种推荐方法都使用的列表
#***********************************************************







