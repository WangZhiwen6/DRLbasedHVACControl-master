from collections import deque
import random
from pprint import pprint
import csv

# 定义一个回放缓冲区类，包括了一些与hvac相关的函数
class ReplayBuffer():
    # 初始化回放缓冲区，设置最大长度
    def __init__(self, max_length):
        self.deque = deque(maxlen=max_length)

    # 添加经验到回放缓冲区
    def add(self, exp:tuple):
        self.deque.append(exp)

    # 从回放缓冲区中随机采样
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.deque, batch_size))
        return state, action, reward, next_state, done

    # 返回回放缓冲区的大小
    def size(self):
        return len(self.deque)


# 定义HVAC动作映射函数
def HVAC_action_map():
    HVAC_action_map = []
    # 遍历所有可能的HVAC动作
    for clg_standard in range(23, 26):
        clg1 = clg_standard
        for clg2 in range(23, 26):
            for clg3 in range(23, 26):
                for clg4 in range(23, 26):
                    for clg5 in range(23, 26):
                        for clg6 in range(23, 26):
                            HVAC_action_map.append([clg1, clg2, clg3, clg4, clg5, clg6])
    return HVAC_action_map

# 定义HVAC设置值函数
def HVAC_setting_value(on_of):
    # 根据HVAC状态设置温度值
    if on_of == 0:
        # temp_setting = [22, 28]
        temp_setting = [22, 28]
    elif on_of == 1:
        # temp_setting = [24, 26]
        temp_setting = [24, 26]
    # elif on_of == 2:
    #     temp_setting = [22, 24]
    # else:
    #     temp_setting = [22, 26]
    return temp_setting

# 将数据保存到CSV文件中
def save_to_csv(DATA):
    csvdir = './data.csv'
    header = [
        'Site Outdoor Air Drybulb Temperature',
        'Zone Windows Total Heat Gain Energy',
        'Zone Air Relative Humidity',
        'Zone Mechanical Ventilation Mass Flow',
        'Zone Thermostat Cooling Setpoint Temperature',
        'Electricity_Zone_1'
    ]
    with open(csvdir, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        data = list(zip(DATA.Site_Outdoor_Air_Drybulb_Temperature,
                        DATA.Zone_Windows_Total_Heat_Gain_Energy_1,
                        DATA.Zone_Air_Relative_Humidity_1,
                        DATA.Zone_Mechanical_Ventilation_Mass_1,
                        DATA.Zone_Thermostat_Cooling_Setpoint_Temperature_1,
                        DATA.Electricity_Zone_1
                        ))
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    map = HVAC_action_map()
    pprint(map)
    print(len(map))
    # save_to_csv()