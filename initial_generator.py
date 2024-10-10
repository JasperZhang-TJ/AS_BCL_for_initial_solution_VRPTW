import numpy as np
from datetime import datetime
from bcl import BCL_Cluster
import math
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class Route(object):
    def __init__(self, depot, init_customer, distance_dic, travelTime_dic, vehicle_info, is_round):
        self.distance_dic = distance_dic
        self.travelTime_dic = travelTime_dic
        self.depot = depot
        self.init_customer = init_customer
        self.init_customer['arrive_time'] = 0 + self.travelTime_dic[(depot['Depot_0']['name'],init_customer['name'] )]
        self.init_customer['end_time'] = self.init_customer['arrive_time'] + self.init_customer['service_time']
        self.is_round = is_round
        if is_round:
            self.route = [self.depot['Depot_0'], init_customer, self.depot['Depot_0']]  #这里初始化回程路径
        else:
            self.route = [self.depot['Depot_0'], init_customer]  # 这里初始化非回程路径
        self.arrival_times = []
        self.travel_and_update_times()
        self.capacity_volume = vehicle_info['asset_capacity_quantity']
        print('debug')


    def insert_customer(self, customer, position):
        # 在路径的 position 位置 插上 customer
        self.route.insert(position, customer)
        feasible, arrival_times = self.travel_and_update_times()
        if not feasible:
            raise Exception("插入后路线不可行")


    # 每次更新的计算复杂度是 n
    def travel_and_update_times(self):
        current_time = 0
        arrival_times = []
        service_times = []
        for i in range(1, len(self.route)):
            prev_customer = self.route[i - 1]
            customer = self.route[i]

            travel_time = self.travelTime_dic[(prev_customer['name'], customer['name'])]
            arrival_time = current_time + travel_time

            # 执行时间窗口限制
            tw_start, tw_end = customer.get('time_window', (0, float('inf')))
            if arrival_time < tw_start:
                service_start_time = tw_start
            else:
                service_start_time = arrival_time
            if arrival_time > tw_end:
                return False, None

            # 添加服务时间
            service_time = customer.get('service_time', 0)
            current_time = service_start_time + service_time
            service_end_time = current_time
            arrival_times.append(arrival_time)
            service_times.append((service_start_time, service_end_time))

            # 更新客户的到达和结束时间
            customer['arrive_time'] = arrival_time
            customer['end_time'] = current_time
            customer['service_period'] = (service_start_time, service_end_time)

        # 对于非回程路线，可能需要在此添加对总行程时间或其他限制的检查
        # 如果有最大行驶时间限制，需要在此处检查 current_time 是否超过限制

        self.arrival_times = arrival_times
        return True, arrival_times


class HeuristicInitial:
    def __init__(self, depot, shipments, distance_dic, travelTime_dic, vehicle_info):
        self.depot = depot
        self.shipments = shipments
        self.cus_inallocated = list(shipments.keys())
        self.cus_allocated = []
        self.distance_dic = distance_dic
        self.travelTime_dic = travelTime_dic
        self.vehicle_info = vehicle_info
        self.routes = []
        self.is_round = vehicle_info['is_round']

        # 获取仓库位置
        depot_location = self.depot['Depot_0']['location']

        # 计算所有客户与仓库的距离，并按距离从小到大排序
        self.sorted_customers = sorted(
            self.cus_inallocated,
            key=lambda cid: self.distance_dic[(depot['Depot_0']['name'], shipments[cid]['name'])]
        )


        # 以下是在实例化 BCL 以方便后续聚类调用
        # pass_shipments = {key: shipment['location'] for key, shipment in shipments.items()}
        # self.BCL = BCL_Cluster(pass_shipments)
        # BCL聚类 以及 链表后处理
        # self.BCL.clustering()
        # self.BCL.plot_customers()
        # self.cluster_members = {cluster_center:info['mem'] for cluster_center,info in self.BCL.cluster_centers.items()}
        # cluster_members = {clus_0:[cus_1, cus_2 ···], ···}
        # self.cus_cluster_dic = {mem: cluster for cluster, members in self.cluster_members.items() for mem in members}
        # cus_cluster_dic = {cus_1:clus_0, cus_2:clus_0, ···}

        # 将时间窗口转换为从参考时间开始的小时数
        reference_time = datetime(2018, 6, 10, 0, 0)

        for shipment in self.shipments.values():
            start_time = shipment['time_window'][0]
            end_time = shipment['time_window'][1]
            # 转换为从reference_time开始的小时数
            start_time_hours = (start_time - reference_time).total_seconds() / 3600
            end_time_hours = (end_time - reference_time).total_seconds() / 3600
            shipment['time_window'] = (start_time_hours, end_time_hours)
            # 确保service_time以小时为单位
            shipment['service_time'] = shipment.get('service_time', 0) / 60  # 如果需要，将分钟转换为小时

    def calculate_insertion_cost(self, route, customer, position):
        prev_customer = route.route[position - 1]
        # 检查是否有下一个客户
        if position < len(route.route):
            next_customer = route.route[position]
            distance_prev_new = self.distance_dic[(prev_customer['name'], customer['name'])]
            distance_new_next = self.distance_dic[(customer['name'], next_customer['name'])]
            distance_prev_next = self.distance_dic[(prev_customer['name'], next_customer['name'])]
            additional_distance = distance_prev_new + distance_new_next - distance_prev_next
        else:
            # 插入到路线末尾，没有下一个客户
            distance_prev_new = self.distance_dic[(prev_customer['name'], customer['name'])]
            additional_distance = distance_prev_new

        # 检查可行性并计算等待时间
        temp_route = route.route[:position] + [customer] + route.route[position:]
        feasible, arrival_times, waiting_time = self.check_and_update(temp_route, route)
        if feasible:
            return additional_distance, waiting_time, True
        else:
            return None, None, False


    # 检查新路径是否可行 计算复杂度为 n
    def check_and_update(self, temp_route, route):
        current_time = 0  # 从仓库开始，时间为零
        total_demand = 0  # 累加客户需求量
        arrival_times = []
        total_waiting_time = 0  # 累计等待时间
        for i in range(1, len(temp_route)):
            prev_customer = temp_route[i - 1]
            customer = temp_route[i]

            # 累加需求量
            total_demand += customer.get('demand', 0)

            # 检查容量限制
            if total_demand > route.capacity_volume:
                return False, None, None  # 超过容量限制

            # 计算旅行时间
            travel_time = self.travelTime_dic[(prev_customer['name'], customer['name'])]
            arrival_time = current_time + travel_time

            # 执行时间窗口限制
            tw_start, tw_end = customer.get('time_window', (0, float('inf')))
            if arrival_time < tw_start:
                waiting_time = tw_start - arrival_time  # 需要等待
                arrival_time = tw_start  # 等待时间窗口打开
                total_waiting_time += waiting_time  # 累计等待时间
            else:
                waiting_time = 0  # 不需要等待

            if arrival_time > tw_end:
                return False, None, None  # 时间窗口违反

            # 添加服务时间
            service_time = customer.get('service_time', 0)
            current_time = arrival_time + service_time
            arrival_times.append(arrival_time)

        # 如果需要，检查返回仓库
        return True, arrival_times, total_waiting_time

    def search_initial_feasible(self):
        while self.sorted_customers:
            # 取最近的客户
            closest_customer_id = self.sorted_customers.pop(0)
            init_customer = self.shipments[closest_customer_id]

            # 创建一个包含该客户的新路线
            new_route = Route(self.depot, init_customer, self.distance_dic, self.travelTime_dic, self.vehicle_info,
                              self.is_round)
            self.routes.append(new_route)
            self.cus_allocated.append(closest_customer_id)

            # 尝试将更多客户插入当前路线
            route_feasible = True  # 保证路径不要违背时间窗的参数
            while route_feasible and self.sorted_customers:
                insertion_options = []
                for customer_id in self.sorted_customers:
                    customer = self.shipments[customer_id]

                    # 根据是否回程，确定插入位置的范围
                    if self.is_round:
                        insertion_positions = range(1, len(new_route.route))
                    else:
                        # 非回程路线，插入位置可以在路线的末尾
                        insertion_positions = range(1, len(new_route.route) + 1)

                    for position in insertion_positions:
                        # 计算插入成本和等待时间
                        cost_increase, waiting_time, feasible = self.calculate_insertion_cost(new_route, customer,
                                                                                              position)
                        if feasible:
                            insertion_options.append({
                                'customer_id': customer_id,
                                'position': position,
                                'cost_increase': cost_increase,
                                'waiting_time': waiting_time
                            })

                if insertion_options:
                    # 提取所有的cost_increase和waiting_time用于归一化
                    cost_increases = [option['cost_increase'] for option in insertion_options]
                    waiting_times = [option['waiting_time'] for option in insertion_options]

                    # 归一化
                    max_cost_increase = max(cost_increases) if max(cost_increases) != 0 else 1
                    max_waiting_time = max(waiting_times) if max(waiting_times) != 0 else 1

                    for option in insertion_options:
                        # 归一化的成本和等待时间
                        normalized_cost = option['cost_increase'] / max_cost_increase
                        normalized_waiting = option['waiting_time'] / max_waiting_time

                        # 计算综合成本
                        total_cost = 1 * normalized_cost + 0.3 * normalized_waiting
                        option['total_cost'] = total_cost

                    # 找到综合成本最小的插入
                    best_option = min(insertion_options, key=lambda x: x['total_cost'])

                    # 将客户插入最佳位置
                    customer_id = best_option['customer_id']
                    position = best_option['position']
                    customer = self.shipments[customer_id]
                    new_route.insert_customer(customer, position)
                    self.sorted_customers.remove(customer_id)
                    self.cus_allocated.append(customer_id)
                else:
                    # 未找到可行的插入，转到下一条路线
                    route_feasible = False

    def calculate_total_cost(self):
        total_cost = 0

        for route in self.routes:
            # 获取当前车辆的固定成本和每单位距离的行驶费用
            vehicle_info = self.vehicle_info
            fixed_cost = vehicle_info['fixed_cost']
            unit_distance_cost = vehicle_info['unit_distance_cost']

            # 计算每条路线的行驶距离
            route_distance = 0
            for i in range(1, len(route.route)):
                prev_customer = route.route[i - 1]
                customer = route.route[i]
                route_distance += self.distance_dic[(prev_customer['name'], customer['name'])]

            # 当前路径的总成本 = 固定成本 + 路线距离 * 每单位距离费用
            route_cost = fixed_cost + route_distance * unit_distance_cost
            total_cost += route_cost

            # 打印每条路径的总成本
            print(f"Route {self.routes.index(route) + 1} cost: {route_cost}")

        # 打印所有路径的总成本
        print(f"Total cost for all routes: {total_cost}")
        return total_cost


# -----------------------------------------以下是调试代码-------------------------------------------------

# 绘制路径函数
def plot_routes(routes, depot):
    plt.figure(figsize=(10, 8))

    # 定义颜色列表，不同路径用不同颜色
    colors = plt.cm.get_cmap('tab10', len(routes))

    # 绘制每一条路径
    for idx, route_single in enumerate(routes, start=1):
        # 获取路径中每个点的坐标
        route_coords = [customer['location'] for customer in route_single.route]

        # 将坐标拆分为经度和纬度列表
        x_coords, y_coords = zip(*route_coords)

        # 使用不同的颜色绘制路径
        plt.plot(x_coords, y_coords, '-o', color=colors(idx-1), label=f"Route {idx}")

    # 标记仓库位置
    depot_coords = depot['Depot_0']['location']
    plt.scatter(*depot_coords, color='red', label='Depot', s=100)

    # 显示图例
    plt.legend()

    # 添加标签和标题
    plt.title("Routes Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # 显示图形
    plt.show()

# 随机生成 分簇客户 以及 客户对应时间窗口的函数
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
            # 生成时间窗
            start_time = np.random.randint(0, 1000)  # 开始时间在[0, 1000]之间随机生成
            end_time = start_time + np.random.randint(30, 120)  # 结束时间为开始时间加上30到120的随机服务时间
            cus_data[f'Customer_{customer_id}'] = {
                'location': (point[0], point[1]),
                'time_window': (start_time, end_time)
            }
            customer_id += 1

    # 如果有剩余客户未分配，继续为他们生成数据
    while customer_id <= num_customers:
        random_cluster = np.random.randint(0, num_clusters)
        point = np.random.multivariate_normal(cluster_centers[random_cluster], cov_matrix)
        start_time = np.random.randint(0, 1000)
        end_time = start_time + np.random.randint(30, 120)
        cus_data[f'Customer_{customer_id}'] = {
            'location': (point[0], point[1]),
            'time_window': (start_time, end_time)
        }
        customer_id += 1

    return cus_data

def round_decimal(data, num=2):
    if isinstance(data, str):
        data = float(data)
    return round(data, num)

def distance_on_unit_sphere(coords1, coords2, factor=17):

    R = 6371.393  # 地球平均半径，单位为千米
    lat1, lon1 = coords1
    lat2, lon2 = coords2
    lat1 = round_decimal(lat1, 8)
    lon1 = round_decimal(lon1, 8)
    lat2 = round_decimal(lat2, 8)
    lon2 = round_decimal(lon2, 8)
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c * (1 + factor * 0.01)
    return d

# 以下是从 119 案例直接写在下面的数据
depot = {
  'Depot_0':{'name': '66cc60f08d7695021861a534',
  'location': (39.66, -104.83),}
}

shipments = {
    'Customer_1': {
        'name': '66cc60f08d7695021861a519-1',
        'location': (42.83300437, -108.7325985),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_2': {
        'name': '66cc60f08d7695021861a529-2',
        'location': (44.52321128, -109.0571007),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_3': {
        'name': '66cc60f08d7695021861a49f-3',
        'location': (33.53997988, -112.0699917),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_4': {
        'name': '66cc60f08d7695021861a491-4',
        'location': (35.18987917, -114.0522221),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_5': {
        'name': '66cc60f08d7695021861a457-5',
        'location': (40.03844627, -105.246093),
        'demand': 110.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_6': {
        'name': '66cc60f08d7695021861a489-6',
        'location': (32.95037762, -112.7246546),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_7': {
        'name': '66cc60f08d7695021861a459-7',
        'location': (39.09385276, -108.5499998),
        'demand': 110.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_8': {
        'name': '66cc60f08d7695021861a51b-8',
        'location': (44.75867495, -108.7584367),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_9': {
        'name': '66cc60f08d7695021861a511-9',
        'location': (41.86750775, -103.6606859),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_10': {
        'name': '66cc60f08d7695021861a449-10',
        'location': (38.2803882, -104.6300066),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_11': {
        'name': '66cc60f08d7695021861a513-11',
        'location': (42.82791424, -103.0030774),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_12': {
        'name': '66cc60f08d7695021861a49b-12',
        'location': (35.19809572, -111.6505083),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_13': {
        'name': '66cc60f08d7695021861a49d-13',
        'location': (32.20499676, -110.8899862),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_14': {
        'name': '66cc60f08d7695021861a4d5-14',
        'location': (39.59979087, -110.8100169),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_15': {
        'name': '66cc60f08d7695021861a4d7-15',
        'location': (37.67742759, -113.061094),
        'demand': 60.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_16': {
        'name': '66cc60f08d7695021861a47b-16',
        'location': (33.58194114, -112.1958238),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_17': {
        'name': '66cc60f08d7695021861a50b-17',
        'location': (42.02871238, -97.43359827),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_18': {
        'name': '66cc60f08d7695021861a52d-18',
        'location': (41.14000694, -104.8197107),
        'demand': 60.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_19': {
        'name': '66cc60f08d7695021861a4cf-19',
        'location': (37.04738853, -112.5254936),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_20': {
        'name': '66cc60f08d7695021861a485-20',
        'location': (35.14817629, -114.5674878),
        'demand': 60.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_21': {
        'name': '66cc60f08d7695021861a4c9-21',
        'location': (39.71027508, -111.8354841),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_22': {
        'name': '66cc60f08d7695021861a517-22',
        'location': (41.24000083, -96.00999007),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_23': {
        'name': '66cc60f08d7695021861a4cd-23',
        'location': (37.84253379, -112.8272065),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_24': {
        'name': '66cc60f08d7695021861a52b-24',
        'location': (42.86661989, -106.3124878),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_25': {
        'name': '66cc60f08d7695021861a515-25',
        'location': (40.81997479, -96.68000086),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_26': {
        'name': '66cc60f08d7695021861a4d9-26',
        'location': (40.45539756, -109.5280022),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_27': {
        'name': '66cc60f08d7695021861a4e1-27',
        'location': (40.7750163, -111.9300519),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_28': {
        'name': '66cc60f08d7695021861a499-28',
        'location': (31.35864016, -109.5483627),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_29': {
        'name': '66cc60f08d7695021861a44f-29',
        'location': (38.54476483, -106.92829),
        'demand': 150.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_30': {
        'name': '66cc60f08d7695021861a4c7-30',
        'location': (38.77247703, -112.0832984),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_31': {
        'name': '66cc60f08d7695021861a495-31',
        'location': (32.68527753, -114.6236084),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_32': {
        'name': '66cc60f08d7695021861a503-32',
        'location': (40.70070559, -99.08114628),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_33': {
        'name': '66cc60f08d7695021861a493-33',
        'location': (36.05478762, -112.1385922),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_34': {
        'name': '66cc60f08d7695021861a4db-34',
        'location': (41.23237856, -111.9680341),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_35': {
        'name': '66cc60f08d7695021861a497-35',
        'location': (34.59001914, -112.4477723),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_36': {
        'name': '66cc60f08d7695021861a443-36',
        'location': (39.69585736, -104.808497),
        'demand': 60.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_37': {
        'name': '66cc60f08d7695021861a521-37',
        'location': (43.02816042, -108.3950481),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_38': {
        'name': '66cc60f08d7695021861a4dd-38',
        'location': (37.10415509, -113.583336),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_39': {
        'name': '66cc60f08d7695021861a525-39',
        'location': (44.28317425, -105.5052503),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_40': {
        'name': '66cc60f08d7695021861a50f-40',
        'location': (41.13980023, -102.9782727),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_41': {
        'name': '66cc60f08d7695021861a51d-41',
        'location': (41.51455772, -109.4649827),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_42': {
        'name': '66cc60f08d7695021861a507-42',
        'location': (42.10139528, -102.8701915),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_43': {
        'name': '66cc60f08d7695021861a447-43',
        'location': (40.56068829, -105.0588693),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_44': {
        'name': '66cc60f08d7695021861a48b-44',
        'location': (31.71314048, -110.066884),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_45': {
        'name': '66cc60f08d7695021861a455-45',
        'location': (40.51728009, -107.5503968),
        'demand': 110.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_46': {
        'name': '66cc60f08d7695021861a483-46',
        'location': (34.49829348, -114.3082789),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_47': {
        'name': '66cc60f08d7695021861a44b-47',
        'location': (38.08649823, -102.6194058),
        'demand': 150.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_48': {
        'name': '66cc60f08d7695021861a445-48',
        'location': (40.41919822, -104.739974),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_49': {
        'name': '66cc60f08d7695021861a45b-49',
        'location': (38.86296246, -104.7919863),
        'demand': 120.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_50': {
        'name': '66cc60f08d7695021861a4d3-50',
        'location': (38.57370363, -109.5491895),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_51': {
        'name': '66cc60f08d7695021861a441-51',
        'location': (39.54658999, -107.3247),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_52': {
        'name': '66cc60f08d7695021861a453-52',
        'location': (38.47727541, -107.8655197),
        'demand': 120.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_53': {
        'name': '66cc60f08d7695021861a44d-53',
        'location': (37.17133445, -104.5063965),
        'demand': 150.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_54': {
        'name': '66cc60f08d7695021861a48d-54',
        'location': (32.25321088, -109.8313945),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_55': {
        'name': '66cc60f08d7695021861a4df-55',
        'location': (40.24889854, -111.63777),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_56': {
        'name': '66cc60f08d7695021861a48f-56',
        'location': (33.69234784, -111.8680402),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_57': {
        'name': '66cc60f08d7695021861a487-57',
        'location': (35.28470542, -110.7006954),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_58': {
        'name': '66cc60f08d7695021861a47d-58',
        'location': (32.83382143, -109.7068801),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_59': {
        'name': '66cc60f08d7695021861a4d1-59',
        'location': (37.87178265, -109.3421995),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_60': {
        'name': '66cc60f08d7695021861a451-60',
        'location': (37.27564333, -107.8799891),
        'demand': 120.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_61': {
        'name': '66cc60f08d7695021861a45d-61',
        'location': (39.73918805, -104.984016),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_62': {
        'name': '66cc60f08d7695021861a481-62',
        'location': (33.42391461, -111.7360844),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 10, 16, 0),
            datetime(2018, 6, 11, 16, 0)
        )
    },
    'Customer_63': {
        'name': '66cc60f08d7695021861a505-63',
        'location': (40.92226829, -98.35798629),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_64': {
        'name': '66cc60f08d7695021861a51f-64',
        'location': (41.7906649, -107.234292),
        'demand': 70.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 13, 16, 0),
            datetime(2018, 6, 14, 16, 0)
        )
    },
    'Customer_65': {
        'name': '66cc60f08d7695021861a47f-65',
        'location': (32.87937421, -111.7566258),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_66': {
        'name': '66cc60f08d7695021861a523-66',
        'location': (43.64597801, -108.2146715),
        'demand': 40.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 14, 16, 0),
            datetime(2018, 6, 15, 16, 0)
        )
    },
    'Customer_67': {
        'name': '66cc60f08d7695021861a509-67',
        'location': (40.20559369, -100.6261683),
        'demand': 90.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    },
    'Customer_68': {
        'name': '66cc60f08d7695021861a4cb-68',
        'location': (41.73593955, -111.8335979),
        'demand': 50.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_69': {
        'name': '66cc60f08d7695021861a50d-69',
        'location': (41.13628623, -100.7705005),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 11, 16, 0),
            datetime(2018, 6, 12, 16, 0)
        )
    },
    'Customer_70': {
        'name': '66cc60f08d7695021861a527-70',
        'location': (41.31136599, -105.5905681),
        'demand': 30.0,
        'service_time': 0,
        'time_window': (
            datetime(2018, 6, 12, 16, 0),
            datetime(2018, 6, 13, 16, 0)
        )
    }
}

# 生成距离字典 {(出发地，目的地): 距离 ···} 以及 运输时间字典{(出发地，目的地): 占用运输时间 ···}
distance_dic = {}
asset_speed = 55.0
vehicles_info = {'66cc60f08d7695021861a53a': {'asset_number': 100, 'asset_capacity_quantity': 500.0, 'asset_capacity_weight': 120000, 'asset_capacity_volume': 1000, 'asset_speed': 55.0, 'fixed_cost': 0, 'unit_distance_cost': 1.2, 'is_round':False},
                '66cc60f08d7695021861a53c': {'asset_number': 100, 'asset_capacity_quantity': 200.0, 'asset_capacity_weight': 120000, 'asset_capacity_volume': 1000, 'asset_speed': 55.0, 'fixed_cost': 0, 'unit_distance_cost': 2.4}, 'is_round':False}
travelTime_dic = {}

# 将所有站点（包括depot和shipments）合并
locations = {**depot, **shipments}

# 使用两两组合计算站点之间的距离
for (key1, data1), (key2, data2) in combinations(locations.items(), 2):
    location1 = data1['location']
    name1 = data1['name']
    location2 = data2['location']
    name2 = data2['name']
    # 计算两点之间的距离
    distance = distance_on_unit_sphere(location1, location2)
    travel_time = distance/asset_speed
    # 存储距离在字典中
    distance_dic[(name1, name2)] = distance
    travelTime_dic[(name1, name2)] = travel_time
    # 反向也存储一份
    distance_dic[(name2, name1)] = distance
    travelTime_dic[(name2, name1)] = travel_time

heuristicInitial = HeuristicInitial(depot, shipments, distance_dic, travelTime_dic, vehicles_info['66cc60f08d7695021861a53a'])

# 记录开始时间
start_time = datetime.now()

heuristicInitial.search_initial_feasible()

# 记录结束时间
end_time = datetime.now()

# 计算初始解生成所花费的时间
elapsed_time = end_time - start_time
print(f"Initial solution generated in: {elapsed_time}")

# 遍历每条路线
for idx, route_single in enumerate(heuristicInitial.routes, start=1):
    # 提取路线中的站点名称
    route_nodes = [customer['name'] for customer in route_single.route]
    # 打印路线编号和经过的站点
    print(f"Route {idx}: {route_nodes}")

# 调用绘图函数，显示所有路径
plot_routes(heuristicInitial.routes, depot)

# 计算并输出总成本
total_cost = heuristicInitial.calculate_total_cost()
