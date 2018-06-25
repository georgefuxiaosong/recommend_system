# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 23:41:44 2018

@author: shentanyue
"""
import pandas as pd
import numpy as np
import math
import time
import datetime
import os
from numpy import linalg as la

os.chdir(r'C:\Users\shentanyue\Desktop\ml-1m')
#载入数据
def loadData():
    rnames=['UserID','MovieID','Rating','Timestamp']
    all_ratings=pd.read_table('ratings.dat',sep='::',header=None
                             ,names=rnames,nrows=300000)
    #sep是标识符，names是标签，nrows是读取的行数
    return all_ratings

#对读取的数据进行分析
def data_explore():
    UR=ratings.groupby([ratings['UserID'],ratings['MovieID']])
    len(UR.size)
    #暂留


#计算每部电影的平均打分
def avg_Rating(ratings):
    movies_mean=ratings['Rating'].groupby(ratings['MovieID']).mean()
    #上面计算所有用户对电影X的平均打分，基于pandas聚合和分组运算的groupby
    movies_id=movies_mean.index
    movies_avg_rating=movies_mean.values
    return movies_id,movies_avg_rating,movies_mean

#计算电影的相似度矩阵
def calculatePC(ratings):
    movies_id,movies_avg_rating,movies_mean=avg_Rating(ratings)
    #avg_Rating函数的调用
    pc_dic={}
    top_movie=len(movies_id)
    for i in range(0,top_movie):
        for j in range(i+1,top_movie):
            movieAID=movies_id[i]
            movieBID=movies_id[j]
            see_moviesA_user=ratings['UserID'][ratings['MovieID']==movieAID]
            see_moviesB_user=ratings['UserID'][ratings['MovieID']==movieBID]
            join_user=np.intersect1d(
                    see_moviesA_user.values,see_moviesB_user.values)
            #同时给电影A,B评分的用户
            movieA_avg=movies_mean[movieAID]
            movieB_avg=movies_mean[movieBID]
            key1=str(movieAID)+':'+str(movieBID)
            key2=str(movieBID)+':'+str(movieAID)
            #value=twoMoviesPC(join_user,movieAID,movieBID,movieA_avg,movieB_avg,ratings)
            #皮尔逊相似度计算
            value=twoMoviesCosSim(join_user,movieAID,movieBID,ratings)
            #余弦相似度计算
            #value=twoMoviesEuclidSim(join_user,movieAID,movieBID,ratings)
            #欧式距离相似度计算
            pc_dic[key1]=value
            pc_dic[key2]=value
    return pc_dic
 
#计算电影A与电影B的相似度，采用皮尔森相似度
#这里利用的是用户的评分
def twoMoviesPC(join_user,movieAID,movieBID,movieA_avg,movieB_avg,ratings):
    cent_AB_sum=0.0#相似度分子
    centA_sum=0.0#相似度分母
    centB_sum=0.0#相似度分母
    movieAB_pc=0.0#电影A,B的相似度
    count=0
    for u in range(len(join_user)):
        count=count+1
        ratA=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieAID].values[0]#用户给电影A评分
        ratB=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieBID].values[0]#用户给电影B评分
        cent_AB=(ratA-movieA_avg)*(ratB-movieB_avg)#去均值中心化
        centA_square=(ratA-movieA_avg)*(ratA-movieA_avg)#去均值平方
        centB_square=(ratB-movieB_avg)*(ratB-movieB_avg)#去均值平方
        cent_AB_sum=cent_AB_sum+cent_AB
        centA_sum=centA_sum+centA_square
        centB_sum=centB_sum+centB_square
    if(centA_sum>0 and centB_sum>0):
        movieAB_pc=cent_AB_sum/math.sqrt(centA_sum*centB_sum)
    return movieAB_pc

#余弦相似度
def twoMoviesCosSim(join_user,movieAID,movieBID,ratings):
    num=0.0 #分子
    denom=0.0#分母
    movieAB_pc=0.0#相似度
    count=0
    for u in range(len(join_user)):
        count=count+1
        ratA=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieAID].values[0]
        ratB=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieBID].values[0]
        num=float(ratA.T*ratB)
        denom=la.norm(ratA)*la.norm(ratB)
        movieAB_pc=0.5+0.5*(num/denom)
    return movieAB_pc

#欧式距离相似度
def twoMoviesEuclidSim(join_user,movieAID,movieBID,ratings):
    movieAB_pc=0.0#相似度
    count=0
    for u in range(len(join_user)):
        count=count+1
        ratA=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieAID].values[0]
        ratB=ratings['Rating'][ratings['UserID']==join_user[u]][ratings['MovieID']==movieBID].values[0]
        movieAB_pc=1.0/(1.0+la.norm(ratA-ratB))
    return movieAB_pc

"""计算电影类型的相似度
#利用电影的类型矩阵求解相似度"""
def twoMoviesType():
    pass


#日期处理： -15天，然后转换为uinxtime
def timePro(last_rat_time,UserU):
    lastDate= datetime.datetime.fromtimestamp(last_rat_time[UserU]) #unix转为日期
    date_sub15=lastDate+datetime.timedelta(days=-15)#减去15天
    unix_sub15=time.mktime(date_sub15.timetuple())#日期转为unix
    return unix_sub15

#取用户最后一次评分前15天评估的电影进行预测
def getHisRat(ratings,last_rat_time,UserUID):
    unix_sub15= timePro(last_rat_time,UserUID)
    UserU_info=ratings[ratings['UserID']==UserUID][ratings['Timestamp']>unix_sub15]
    return UserU_info

#预测用户U对电影C的打分   
"""----------评分预测法--------------------"""
def hadSeenMovieByUser(UserUID,MovieA,ratings,pc_dic,movies_mean):
    pre_rating=0.0    
    last_rat_time=ratings['Timestamp'].groupby([ratings['UserID']]).max()#获取用户U最近一次评分日期
    UserU_info= getHisRat(ratings,last_rat_time,UserUID)#获取用户U过去看过的电影

    flag=0#表示新电影，用户U是否给电影A打过分
    wmv=0.0#相似度*mv平均打分去均值后之和
    w=0.0#相似度之和
    movie_userU=UserU_info['MovieID'].values#当前用户看过的电影
    if MovieA in movie_userU:
        flag=1
        pre_rating=UserU_info['Rating'][UserU_info['MovieID']==MovieA].values
    else:
        for mv in movie_userU:
            key=str(mv)+':'+str(MovieA)
            rat_U_mv=UserU_info['Rating'][UserU_info['MovieID']==mv][UserU_info['UserID']==UserUID].values#用户U对看过电影mv的打分
            wmv=(wmv+pc_dic[key]*(rat_U_mv-movies_mean[mv]))#相似度*mv平均打分去均值后之和
            w=(w+pc_dic[key])#看过电影与新电影相似度之和
            #print ('---have seen mv %d with new mv %d,%f,%f'%(mv,MovieA,wmv,w))            
        pre_rating=(movies_mean[MovieA]+wmv/w)
    print ('\n\n-flag:%d---User:%d rating movie:%d with %f score----\n\n' %(flag,UserUID,MovieA,pre_rating))
    return pre_rating,flag


"""----------测试--------------"""
if __name__=='__main__':
    all_ratings=loadData()
    movie_num=100#控制电影数，只针对电影ID在该范围的数据进行计算，否则数据量太大    
    ratings=all_ratings[all_ratings['MovieID']<=movie_num] 

    movies_id,movies_avg_rating,movies_mean=avg_Rating(ratings)
    pc_dic=calculatePC(ratings)#电影相似度矩阵
    movie_sort=[]
    movie_index=[]
    #-------预测---------
    UserUID=10
    for MovieA in range(1,50): #有些电影编号缺失该如何处理???
        pre_rating,flag=hadSeenMovieByUser(UserUID,MovieA,ratings,pc_dic,movies_mean)
        if flag == 0:
                movie_index=[(MovieA,pre_rating)]
                movie_sort.extend(movie_index)
    movie_sort.sort(key=lambda x:x[1],reverse=True)
    print(movie_sort)

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            