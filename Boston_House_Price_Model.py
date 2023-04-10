# By-StarWing
# 2023.4.10

import csv
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import copy

# :Number of Instances: 506
# :Number of Attributes: 14

# 属性描述（不一定对..）: Attribute Information (in order):
#         - CRIM     犯罪率；per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    非零售商业用地占比；proportion of non-retail business acres per town
#         - CHAS     是否临Charles河；Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      氮氧化物浓度；nitric oxides concentration (parts per 10 million)
#         - RM       房屋房间数；average number of rooms per dwelling
#         - AGE      房屋年龄；proportion of owner-occupied units built prior to 1940
#         - DIS      和就业中心的距离；weighted distances to five Boston employment centres
#         - RAD      是否容易上高速路；index of accessibility to radial highways
#         - TAX      税率；full-value property-tax rate per $10,000
#         - PTRATIO  学生人数比老师人数；pupil-teacher ratio by town
#         - B        城镇黑人比例计算的统计值；1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
#         - LSTAT    低收入人群比例；% lower status of the population
#         - MEDV     房价中位数；Median value of owner-occupied homes in $1000's

# 设定训练集大小
DATA_NUM = 500
# 梯度下降步长
A = 0.000005
# 梯度下降次数
ITER_NUM = 100000

w_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])  # 初始模型w
b = 0  # 初始模型b


def read_model():
    save_w_vector = []
    save_b = 0
    save_model_info1 = []
    with open('model.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            save_model_info1.append(row)
    for i in range(13):
        save_w_vector.append(float(save_model_info1[2][i]))
    save_b = float(save_model_info1[2][13])
    return np.array(save_w_vector), save_b


def read_csv():
    info = []
    with open("boston_house_prices.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(row)
    return info


# 获取数据集大小
def get_data_num(info):
    return int(info[0][0])


# 获取模型特征数量
def get_vector_num(info):
    return int(info[0][1])


# 获取训练集(num小于等于数据集大小)
def get_training(info, num):
    training = []
    for i in range(2, num + 2):
        training.append(list(map(float, info[i])))
    return np.array(training)


# 获取输入特征集
def get_x_vector_list(training, vector_num):
    x_vector_list = []
    for vector in training:
        x_vector_list.append(np.delete(vector, [vector_num]))
    return np.array(x_vector_list)


# 获取输出目标集
def get_y_list(training, vector_num):
    y_list = []
    for vector in training:
        y_list.append(vector[vector_num])
    return y_list


# 返回一个随机样本
def get_rand(training, vector_num):
    i = random.randint(0, len(training))
    return np.delete(training[i], [vector_num]), training[i][vector_num]


# 模型
def model(x_vector):
    return np.dot(w_vector, x_vector) + b


# 代价函数
def cost_function(x_vector_list, y_list):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + math.pow(model(x_vector_list[i]) - y_list[i], 2)
    sum = sum / (DATA_NUM * 2)
    return sum


# 代价函数对w的偏导数:
def cost_w(x_vector_list, y_list, j):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + (model(x_vector_list[i]) - y_list[i]) * x_vector_list[i][j]
    sum = sum / DATA_NUM
    return sum


# 代价函数对b的偏导数:
def cost_b(x_vector_list, y_list):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + (model(x_vector_list[i]) - y_list[i])
    sum = sum / DATA_NUM
    return sum


# 梯度下降算法(输入特征x, 输出目标y)
def gradient_descent(x_vector_list, y_list):
    global b, w_vector
    cost_x = []
    cost_y = []
    last_cost = 9999999999
    cost = cost_function(x_vector_list, y_list)
    for i in range(ITER_NUM):
        b_temp = b - A * cost_b(x_vector_list, y_list)
        w_vector_temp = copy.deepcopy(w_vector)
        for j in range(w_vector.size):
            w_vector_temp[j] = w_vector[j] - A * cost_w(x_vector_list, y_list, j)
        w_vector = w_vector_temp
        b = b_temp
        last_cost = cost
        cost = cost_function(x_vector_list, y_list)
        # if cost < 10000:
        #     cost_x.append(i)
        #     cost_y.append(cost)
        print("第%d次迭代cost： %f" % (i, cost))
        if last_cost - cost < 0.0001:
            break
    # plt.plot(cost_x, cost_y)
    # plt.show()
    return w_vector, b


# 测试模型
def test(x_vector, y):
    print("------------随机测试------------")
    cal_y = model(x_vector)
    print("预期输出: ", y)
    print("计算结果： ", cal_y)
    print("误差： ", y - cal_y)
    print("------------------------------")


def save_model():
    header = ['犯罪率', '25000平英尺以上地块的住宅用地比例', '非零售商业用地占比', '是否临Charles河', '氮氧化物浓度',
              '房屋房间数', '房屋年龄', '和就业中心的距离', '是否容易上高速路', '税率', '学生人数比老师人数',
              '城镇黑人比例计算的统计值', '低收入人群比例', '固定值']
    m = np.append(w_vector, [b])
    with open('model.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(m)
    print('已保存至model.csv')


def main():
    global w_vector, b
    w_vector, b = read_model()
    info = read_csv()
    data_num = get_data_num(info)
    vector_num = get_vector_num(info)
    training = get_training(info, DATA_NUM)
    x_vector, y = get_rand(training, vector_num)
    x_vector_list = get_x_vector_list(training, vector_num)
    y_list = get_y_list(training, vector_num)
    gradient_descent(x_vector_list, y_list)
    print("模型w: ", w_vector)
    print("模型b:", b)
    test(x_vector, y)
    save_model()


main()

