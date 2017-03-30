import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import mysql.connector
import random



error_list=[]

def centroid_initialize(input_data,all_centroid,stop_code,k,cluster_converge):


    centroid_init=[]
    random.seed(random.randrange(1,1000))
    rondom_mat = random.sample(range(len(input_data)),k)
    # rondom_mat=input_data[random.sample(range(len(input_data)),k),1:len(input_data[0])-1]
    for i in range(0,k):

        if list(input_data[rondom_mat[i]][1:9]) not in centroid_init:
            centroid_init.append(list(input_data[rondom_mat[i]][1:9]))
        else:
            j=i
            centroid_init.append(list(input_data[rondom_mat[i]+i][1:9]))
    all_centroid.append(centroid_init)
    return centroid_init

def get_dis(a,b):
    temp_list=[]
    for i in range(0,len(b)):
        diff=pow((a[i+1])-(b[i]),2)
        temp_list.append(diff)
    dis=math.sqrt(sum(temp_list))
    return dis

def assign_points(centroid,input_data,stop_code,k,cluster_converge):
    clustered_data=[]
    list = [[] for c in centroid[0]]

    for row in range(0, len(input_data)):

        min_dist = get_dis(input_data[row], centroid[0][0])
        ind = 0
        for c in range(0, len(centroid[0])):
            distance = get_dis(input_data[row], centroid[0][c])
            if distance < min_dist:
                min_dist = distance
                ind = c
        list[ind].append(input_data[row])
        if len(list) !=0:
            clustered_data.append(list)

    return list


def update_centroids(list,all_centroid,input_data,stop_code,k,cluster_converge):
    centroid=[]

    for i in range(0,len(list)):

            cluster_size=len(list[i])

            temp_mat=np.matrix(list[i]) #Converting list and storing in matrice to get sum of the column

            sum_array=[]
            #need to change this based on columns which are being clustered

            for m in range(1,9):
                sum_array.append(np.sum(temp_mat[:, m]))
            new_sum_arr=[float(x) / float(cluster_size) for x in sum_array]
            centroid.append(new_sum_arr)
    all_centroid.append(centroid)
    return centroid

def check_convergence(centroids,input_data,stop_code,k,cluster_converge):
    cluster_converge=True
    centroid_size = len(centroids)
    diff_sum = []
    for j in range(0, len(centroids)):

        if (j != len(centroids) - 1):
            for n in range(0, len(centroids)):
                diff_sum.append([float(x)-float(y) for x,y in zip(centroids[j][n], centroids[j+1][n])])


    for i in range(0, len(diff_sum)):
                sum_of_centroid =+ (math.sqrt(sum(float(p) * float(q) for p, q in zip(diff_sum[i], diff_sum[i])))/float(k))


    if float(sum_of_centroid) < float(stop_code):
         cluster_converge= False

    return cluster_converge


def get_error(list,centroid):
    total_error=0
    for j in range(0, len(list)):
        b_count=0
        m_count=0
        error_val=0
        for k in range(0,len(list[j])):
            if list[j][k][len(list[j][k])-1] == 2:
                b_count= b_count+1
            else:
                m_count=m_count+1
        if(b_count > m_count):
            error_val=float(m_count)/float(b_count+m_count)

        else:
            error_val = float(b_count) / float(b_count + m_count)

        total_error= float(total_error)+float(error_val)

    all_centroid=[]
    return total_error



def k_means(input_data,all_centroid,stop_code,k,cluster_converge):
    cluster_converge = True
    #created empty clsusters based on the k value
    centroid_initialize(input_data,all_centroid,stop_code,k,cluster_converge)


    while cluster_converge:
        clustered_data=assign_points(all_centroid,input_data,stop_code,k,cluster_converge)
        update_centroids(clustered_data,all_centroid,input_data,stop_code,k,cluster_converge)
        cluster_converge=check_convergence(all_centroid,input_data,stop_code,k,cluster_converge)
        all_centroid.pop(0)

    return get_error(clustered_data,all_centroid)

def main():
    data = []
    cnx = mysql.connector.connect(user='******', password='******', host='localhost',
                                  database='datamining')
    cursor = cnx.cursor()
    query = ("SELECT * FROM delta")
    cursor.execute(query)
    for line in cursor:
        data.append(list(line))

    input_data = np.array(data, np.int32)
    cnx.close()
    all_centroid = []
    stop_code = 10
    k = 5
    clustered_data = []
    cluster_converge = True

    return k_means(input_data,all_centroid,stop_code,k,cluster_converge)


for i in range(1,21):
    if __name__ == '__main__': error_list.append(main())
print 'Summation of 20 iteration :',sum(error_list)
plt.plot(error_list,'ro')
plt.ylabel('Errors of centroid')
plt.show()

