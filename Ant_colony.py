import numpy as np
import datetime
from BCL import BCL_cluster


class ant(object):
    def __init__(self, x, y):
        self.x = x

class ant_colony:
    def __init__(self, depot, shipments):
        self.depot = depot
        self.shipments = shipments

        # 以下是在实例化 BCL 以方便后续聚类调用
        pass_shipments = {key: shipment['location'] for key, shipment in shipments.items()}
        self.BCL = BCL_cluster(pass_shipments)
        # BCL聚类 以及 链表后处理
        self.BCL.clustering()
        self.BCL.plot_customers()
        self.cluster_members = {cluster_center:info['mem'] for cluster_center,info in self.BCL.cluster_centers.items()}
        # cluster_members = {clus_0:[cus_1, cus_2 ···], ···}
        self.cus_cluster_dic = {mem: cluster for cluster, members in self.cluster_members.items() for mem in members}
        # cus_cluster_dic = {cus_1:clus_0, cus_2:clus_0, ···}

    def search_initial_feasible(self):

        print("debug")

# -----------------------------------------以下是调试代码-------------------------------------------------
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

# 以下是从 119 案例直接写在下面的数据
depot = {
  'name': '66cc60f08d7695021861a534',
  'location': (39.66, -104.83),
}

shipments = {
 'Customer_1': {'name': '66cc60f08d7695021861a519-1',
  'location': (42.83300437, -108.7325985),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_2': {'name': '66cc60f08d7695021861a529-2',
  'location': (44.52321128, -109.0571007),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_3': {'name': '66cc60f08d7695021861a49f-3',
  'location': (33.53997988, -112.0699917),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_4': {'name': '66cc60f08d7695021861a491-4',
  'location': (35.18987917, -114.0522221),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_5': {'name': '66cc60f08d7695021861a457-5',
  'location': (40.03844627, -105.246093),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_6': {'name': '66cc60f08d7695021861a489-6',
  'location': (32.95037762, -112.7246546),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_7': {'name': '66cc60f08d7695021861a459-7',
  'location': (39.09385276, -108.5499998),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_8': {'name': '66cc60f08d7695021861a51b-8',
  'location': (44.75867495, -108.7584367),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_9': {'name': '66cc60f08d7695021861a511-9',
  'location': (41.86750775, -103.6606859),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_10': {'name': '66cc60f08d7695021861a449-10',
  'location': (38.2803882, -104.6300066),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_11': {'name': '66cc60f08d7695021861a513-11',
  'location': (42.82791424, -103.0030774),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_12': {'name': '66cc60f08d7695021861a49b-12',
  'location': (35.19809572, -111.6505083),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_13': {'name': '66cc60f08d7695021861a49d-13',
  'location': (32.20499676, -110.8899862),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_14': {'name': '66cc60f08d7695021861a4d5-14',
  'location': (39.59979087, -110.8100169),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_15': {'name': '66cc60f08d7695021861a4d7-15',
  'location': (37.67742759, -113.061094),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_16': {'name': '66cc60f08d7695021861a47b-16',
  'location': (33.58194114, -112.1958238),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_17': {'name': '66cc60f08d7695021861a50b-17',
  'location': (42.02871238, -97.43359827),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_18': {'name': '66cc60f08d7695021861a52d-18',
  'location': (41.14000694, -104.8197107),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_19': {'name': '66cc60f08d7695021861a4cf-19',
  'location': (37.04738853, -112.5254936),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_20': {'name': '66cc60f08d7695021861a485-20',
  'location': (35.14817629, -114.5674878),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_21': {'name': '66cc60f08d7695021861a4c9-21',
  'location': (39.71027508, -111.8354841),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_22': {'name': '66cc60f08d7695021861a517-22',
  'location': (41.24000083, -96.00999007),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_23': {'name': '66cc60f08d7695021861a4cd-23',
  'location': (37.84253379, -112.8272065),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_24': {'name': '66cc60f08d7695021861a52b-24',
  'location': (42.86661989, -106.3124878),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_25': {'name': '66cc60f08d7695021861a515-25',
  'location': (40.81997479, -96.68000086),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_26': {'name': '66cc60f08d7695021861a4d9-26',
  'location': (40.45539756, -109.5280022),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_27': {'name': '66cc60f08d7695021861a4e1-27',
  'location': (40.7750163, -111.9300519),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_28': {'name': '66cc60f08d7695021861a499-28',
  'location': (31.35864016, -109.5483627),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_29': {'name': '66cc60f08d7695021861a44f-29',
  'location': (38.54476483, -106.92829),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_30': {'name': '66cc60f08d7695021861a4c7-30',
  'location': (38.77247703, -112.0832984),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_31': {'name': '66cc60f08d7695021861a495-31',
  'location': (32.68527753, -114.6236084),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_32': {'name': '66cc60f08d7695021861a503-32',
  'location': (40.70070559, -99.08114628),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_33': {'name': '66cc60f08d7695021861a493-33',
  'location': (36.05478762, -112.1385922),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_34': {'name': '66cc60f08d7695021861a4db-34',
  'location': (41.23237856, -111.9680341),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_35': {'name': '66cc60f08d7695021861a497-35',
  'location': (34.59001914, -112.4477723),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_36': {'name': '66cc60f08d7695021861a443-36',
  'location': (39.69585736, -104.808497),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_37': {'name': '66cc60f08d7695021861a521-37',
  'location': (43.02816042, -108.3950481),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_38': {'name': '66cc60f08d7695021861a4dd-38',
  'location': (37.10415509, -113.583336),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_39': {'name': '66cc60f08d7695021861a525-39',
  'location': (44.28317425, -105.5052503),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_40': {'name': '66cc60f08d7695021861a50f-40',
  'location': (41.13980023, -102.9782727),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_41': {'name': '66cc60f08d7695021861a51d-41',
  'location': (41.51455772, -109.4649827),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_42': {'name': '66cc60f08d7695021861a507-42',
  'location': (42.10139528, -102.8701915),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_43': {'name': '66cc60f08d7695021861a447-43',
  'location': (40.56068829, -105.0588693),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_44': {'name': '66cc60f08d7695021861a48b-44',
  'location': (31.71314048, -110.066884),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_45': {'name': '66cc60f08d7695021861a455-45',
  'location': (40.51728009, -107.5503968),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_46': {'name': '66cc60f08d7695021861a483-46',
  'location': (34.49829348, -114.3082789),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_47': {'name': '66cc60f08d7695021861a44b-47',
  'location': (38.08649823, -102.6194058),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_48': {'name': '66cc60f08d7695021861a445-48',
  'location': (40.41919822, -104.739974),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_49': {'name': '66cc60f08d7695021861a45b-49',
  'location': (38.86296246, -104.7919863),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_50': {'name': '66cc60f08d7695021861a4d3-50',
  'location': (38.57370363, -109.5491895),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_51': {'name': '66cc60f08d7695021861a441-51',
  'location': (39.54658999, -107.3247),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_52': {'name': '66cc60f08d7695021861a453-52',
  'location': (38.47727541, -107.8655197),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_53': {'name': '66cc60f08d7695021861a44d-53',
  'location': (37.17133445, -104.5063965),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_54': {'name': '66cc60f08d7695021861a48d-54',
  'location': (32.25321088, -109.8313945),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_55': {'name': '66cc60f08d7695021861a4df-55',
  'location': (40.24889854, -111.63777),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_56': {'name': '66cc60f08d7695021861a48f-56',
  'location': (33.69234784, -111.8680402),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_57': {'name': '66cc60f08d7695021861a487-57',
  'location': (35.28470542, -110.7006954),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_58': {'name': '66cc60f08d7695021861a47d-58',
  'location': (32.83382143, -109.7068801),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_59': {'name': '66cc60f08d7695021861a4d1-59',
  'location': (37.87178265, -109.3421995),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_60': {'name': '66cc60f08d7695021861a451-60',
  'location': (37.27564333, -107.8799891),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_61': {'name': '66cc60f08d7695021861a45d-61',
  'location': (39.73918805, -104.984016),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_62': {'name': '66cc60f08d7695021861a481-62',
  'location': (33.42391461, -111.7360844),
  'time_window': (datetime.datetime(2018, 6, 10, 16, 0), datetime.datetime(2018, 6, 11, 16, 0))},

 'Customer_63': {'name': '66cc60f08d7695021861a505-63',
  'location': (40.92226829, -98.35798629),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_64': {'name': '66cc60f08d7695021861a51f-64',
  'location': (41.7906649, -107.234292),
  'time_window': (datetime.datetime(2018, 6, 13, 16, 0), datetime.datetime(2018, 6, 14, 16, 0))},

 'Customer_65': {'name': '66cc60f08d7695021861a47f-65',
  'location': (32.87937421, -111.7566258),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_66': {'name': '66cc60f08d7695021861a523-66',
  'location': (43.64597801, -108.2146715),
  'time_window': (datetime.datetime(2018, 6, 14, 16, 0), datetime.datetime(2018, 6, 15, 16, 0))},

 'Customer_67': {'name': '66cc60f08d7695021861a509-67',
  'location': (40.20559369, -100.6261683),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))},

 'Customer_68': {'name': '66cc60f08d7695021861a4cb-68',
  'location': (41.73593955, -111.8335979),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_69': {'name': '66cc60f08d7695021861a50d-69',
  'location': (41.13628623, -100.7705005),
  'time_window': (datetime.datetime(2018, 6, 11, 16, 0), datetime.datetime(2018, 6, 12, 16, 0))},

 'Customer_70': {'name': '66cc60f08d7695021861a527-70',
  'location': (41.31136599, -105.5905681),
  'time_window': (datetime.datetime(2018, 6, 12, 16, 0), datetime.datetime(2018, 6, 13, 16, 0))}

}

ant_colony = ant_colony(depot, shipments)