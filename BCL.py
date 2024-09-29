from re import search

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.random import random
import random  # 需要导入 random 库
from sympy.stats.sampling.sample_numpy import numpy


class BCL_cluster:
    '''
    cus_data 是 客户的经纬度数组 ： [('cus_name',latitude,longitude),...]
    '''
    def __init__(self, cus_data):
        self.cus_data = cus_data
        self.customers = list(cus_data.keys())
        self.judge_angle = 90
        self.delta_L = 1
        self.forgetting_factor = 1/len(self.customers)
        self.learn_alpha = 0.02

        # 提取经度和纬度
        latitudes = [v[0] for v in self.cus_data.values()]
        longitudes = [v[1] for v in self.cus_data.values()]

        self.d_0 = max(max(latitudes)-min(latitudes), max(longitudes)-min(longitudes))/8

        # 分别计算经度和纬度的标准差
        self.lat_std = np.std(latitudes)
        self.lon_std = np.std(longitudes)
        self.lat_mean = np.mean(latitudes)
        self.lon_mean = np.mean(longitudes)

        self.center_count = 0
        self.cluster_centers = {'center'+str(self.center_count):{'lat':self.lat_mean,'lon':self.lon_mean,'mem':[],
                                                                 'l_move':(0,0),'c_move':(0,0),'act_level':0}}

        self.terminate = 0 # 用来判断聚类是否终止的属性值

        self.cus_count = 0

        self.interation_count = 0


    # 定义一个函数，根据输入的种子生成10个不同的随机打乱的数组，并返回合并后的大数组
    def generate_random_arrays_with_merged(self, seed_value):
        original_array = list(range(len(self.customers)))  # 初始数组 [0, 1, 2, ..., 299]
        random.seed(seed_value)  # 根据输入的种子固定随机顺序

        random_arrays = []
        for _ in range(100):  # 生成100个随机数组
            shuffled_array = original_array[:]  # 复制原始数组
            random.shuffle(shuffled_array)  # 随机打乱顺序
            random_arrays.append(shuffled_array)

        # 将10个数组合并成一个大的数组
        merged_array = []
        for array in random_arrays:
            merged_array.extend(array)

        return random_arrays, merged_array


    def clustering(self):
        search_lists,search_sequence = self.generate_random_arrays_with_merged(self.judge_angle)

        search_step = 0

        while self.terminate == 0:
            self.cus_count = search_sequence[search_step]
            search_step += 1

            # 当前客户
            cus_0 = self.customers[self.cus_count]
            cus_data_0 = self.cus_data[cus_0]

            self.interation_count += 1
            print(self.interation_count)
            if self.interation_count >= 100*len(self.customers):
                self.terminate = 1

            # 计算客户与每个簇中心的距离
            min_distance = float('inf')  # 初始化最小距离为无穷大
            closest_center = None  # 初始化最近的中心

            for center_name, center_data in self.cluster_centers.items():
                center_lat = center_data['lat']
                center_lon = center_data['lon']

                # 计算当前客户与该簇中心的距离
                distance = np.sqrt((cus_data_0[0] - center_lat) ** 2 + (cus_data_0[1] - center_lon) ** 2)

                # 找到距离最小的簇中心
                if distance < min_distance:
                    min_distance = distance
                    closest_center = center_name

            # 遍历所有簇中心，减少其激活水平
            for center_name, center_data in self.cluster_centers.items():
                if center_name != closest_center:
                    # 使用 λ 减小激活水平
                    center_data['act_level'] *= (1 - self.forgetting_factor)

            # 获取当前的簇中心信息
            closest_center_data = self.cluster_centers[closest_center]

            # 计算客户点和簇中心的差值向量
            delta_lat = cus_data_0[0] - self.cluster_centers[closest_center]['lat']
            delta_lon = cus_data_0[1] - self.cluster_centers[closest_center]['lon']

            # 计算角度变化
            l_move_vector = np.array(closest_center_data['l_move'])
            c_move_vector = np.array(closest_center_data['c_move'])
            angle_change = np.degrees(np.arccos(
                np.clip(np.dot(l_move_vector, c_move_vector) / (LA.norm(l_move_vector) * LA.norm(c_move_vector) + 1e-6),
                        -1.0, 1.0)))

            # 计算移动距离的变化
            l_move_distance = LA.norm(l_move_vector)
            c_move_distance = LA.norm(c_move_vector)
            distance_change = min(l_move_distance, c_move_distance)

            # 判断是否符合激活条件
            if angle_change > self.judge_angle and distance_change > self.d_0:
                print('activated!!')
                closest_center_data['act_level'] += self.delta_L  # 增加激活水平
            else:
                closest_center_data['act_level'] = max(closest_center_data['act_level'] - self.forgetting_factor * closest_center_data['act_level'],0)  # 衰减激活水平

            # 判断是否超过阈值 L0
            if closest_center_data['act_level'] > self.d_0:
                # 生成新节点
                self.center_count += 1
                new_center_name = 'center' + str(self.center_count)

                # 计算新节点的位置
                new_center_lat = closest_center_data['lat'] + self.learn_alpha * delta_lat
                new_center_lon = closest_center_data['lon'] + self.learn_alpha * delta_lon

                # 添加新节点到簇中心
                self.cluster_centers[new_center_name] = {
                    'lat': new_center_lat,
                    'lon': new_center_lon,
                    'mem': [],
                    'l_move': (0, 0),
                    'c_move': (0, 0),
                    'act_level': 0  # 新节点的初始激活水平为 0
                }

                # 重置原簇中心的激活水平
                closest_center_data['act_level'] = 0

            # 如果没有超过阈值，仅更新簇中心位置 和 移动信息
            else:
                # 更新移动量 c_move 和 l_move
                self.cluster_centers[closest_center]['l_move'] = self.cluster_centers[closest_center]['c_move']
                self.cluster_centers[closest_center]['c_move'] = (delta_lat, delta_lon)
                self.cluster_centers[closest_center]['lat'] += self.learn_alpha * self.cluster_centers[closest_center]['c_move'][0]
                self.cluster_centers[closest_center]['lon'] += self.learn_alpha * self.cluster_centers[closest_center]['c_move'][1]

            # self.cus_count += 1
            # if self.cus_count == len(self.customers):
            #     self.terminate = 0

        # 遍历所有客户，并将每个客户分配到距离最近的簇中心
        for cus_name, cus_data in self.cus_data.items():
            min_distance = float('inf')  # 初始化最小距离为无穷大
            closest_center = None  # 初始化最近的中心

            # 计算客户与每个簇中心的距离
            for center_name, center_data in self.cluster_centers.items():
                center_lat = center_data['lat']
                center_lon = center_data['lon']

                # 计算客户与该簇中心的距离
                distance = np.sqrt((cus_data[0] - center_lat) ** 2 + (cus_data[1] - center_lon) ** 2)

                # 找到距离最小的簇中心
                if distance < min_distance:
                    min_distance = distance
                    closest_center = center_name

            # 将客户添加到最接近的簇中心的成员列表中
            self.cluster_centers[closest_center]['mem'].append(cus_name)


    def plot_customers(self):
        # 定义不同颜色的列表，确保每个簇都有不同的颜色
        colors = plt.cm.get_cmap('tab10', len(self.cluster_centers))  # 使用 'tab10' 调色板，最多10种颜色

        # 创建散点图
        plt.figure(figsize=(8, 6))

        # 遍历所有簇中心及其成员
        for idx, (center_name, center_data) in enumerate(self.cluster_centers.items()):
            # 获取簇中心的颜色
            color = colors(idx)

            # 绘制簇中心的成员
            for cus_name in center_data['mem']:
                cus_data = self.cus_data[cus_name]
                plt.scatter(cus_data[0], cus_data[1], color=color, marker='o')

            # 绘制簇中心本身
            plt.scatter(center_data['lat'], center_data['lon'], color=color, marker='x', s=100,
                        label=f'{center_name} center')

        # 设置图表标题和标签
        plt.title("Customer Locations and Cluster Centers with Members")
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.legend()

        # 显示图表
        plt.grid(True)
        plt.show()


#   ------------------------------------以下是测试代码-------------------------------

def generate_gaussian_cus_data(num_customers, num_clusters):
    cus_data = {}

    # 设置每个簇的均值和协方差矩阵
    cluster_centers = np.random.uniform(-180, 180, (num_clusters, 2))  # 设定簇中心的均值
    cov_matrix = [[220, 0],
                  [0, 220]]  # 设置协方差矩阵，控制数据分散程度

    customer_id = 1
    for i in range(num_clusters):
        # 为每个簇生成一组高斯分布的点
        points = np.random.multivariate_normal(cluster_centers[i], cov_matrix, num_customers // num_clusters)
        for point in points:
            cus_data[f'Customer_{customer_id}'] = (point[0], point[1])
            customer_id += 1

    # 如果有剩余客户未分配，继续为他们生成数据
    while customer_id <= num_customers:
        random_cluster = np.random.randint(0, num_clusters)
        point = np.random.multivariate_normal(cluster_centers[random_cluster], cov_matrix)
        cus_data[f'Customer_{customer_id}'] = (point[0], point[1])
        customer_id += 1

    return cus_data

def main():

    # 生成20个客户，分布在3个聚类中
    num_customers = 300
    num_clusters = 16
    cus_data = generate_gaussian_cus_data(num_customers, num_clusters)

    for i in range(10):
        BCL_test = BCL_cluster(cus_data)
        BCL_test.clustering()

        print(f"纬度标准差: {BCL_test.lat_std}")
        print(f"经度标准差: {BCL_test.lon_std}")

        # 绘制客户的经纬度散点图
        BCL_test.plot_customers()

    print('--------------------------test finished----------------------------')

if __name__ == '__main__':
    main()



