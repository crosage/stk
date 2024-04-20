import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import Topos, load, EarthSatellite, utc
from datetime import datetime, timedelta

# default_time = datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)
ts = load.timescale()


class GroundStation:

    def __init__(self, latitude_degrees, longitude_degrees, name, elevation_meters=0,bandwidth=32000,Xmtr_Power=30,Xmtr_Gain=0):
        self.location = Topos(latitude_degrees=latitude_degrees, longitude_degrees=longitude_degrees, elevation_m=elevation_meters)
        self.name = name
        # 海拔高度
        self.elevation_meters=elevation_meters
        # 带宽
        self.bandwidth=bandwidth
        # Xmtr_Power: 发射功率
        self.Xmtr_Power=Xmtr_Power
        # Xmtr_Gain: 发射增益
        self.Xmtr_Gain=Xmtr_Gain
        # ts = load.timescale()

    def get_position_at_time(self, time):
        if isinstance(time, datetime):
            return self.location.at(ts.utc(time)).position.km
        return self.location.at(time).position.km

    def get_position_over_time(self, start_time, end_time, interval_seconds=60):
        times = []
        start_time = ts.utc(start_time)
        end_time = ts.utc(end_time)
        while start_time < end_time:
            times.append(start_time)
            start_time = start_time + interval_seconds / 86400
        positions = [self.get_position_at_time(t) for t in times]
        return positions


class Satellite:
    default_time = datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)
    def __init__(self, line1, line2, name,Xmtr_Power=30,Xmtr_Gain=0):
        
        self.satellite = EarthSatellite(line1, line2, name)
        self.name = name
        # Xmtr_Power: 发射功率，单位为dBW
        self.Xmtr_Power=Xmtr_Power
        # Xmtr_Gain: 发射增益
        self.Xmtr_Gain=Xmtr_Gain
        self.path = []

    def get_position_at_time(self, time):
        if isinstance(time, datetime):
            return self.satellite.at(ts.utc(time)).position.km
        return self.satellite.at(time).position.km

    def get_position_over_time(self, start_time, end_time, interval_seconds=60):
        times = []
        start_time = ts.utc(start_time)
        end_time = ts.utc(end_time)
        while start_time < end_time:
            times.append(start_time)
            start_time = start_time + interval_seconds / 86400
        # times=ts.utc(start_time,end_time,interval_seconds=interval_seconds)
        # print(times)
        positions = [self.get_position_at_time(t) for t in times]
        return positions

    def get_path(self, time=default_time):
        start_time = time
        end_time = start_time + timedelta(hours=2)
        positions = self.get_position_over_time(start_time, end_time)
        return positions


def read_tle_file(file_path):
    satellites = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        for i in range(0, num_lines, 3):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            satellites[name] = Satellite(line1,line2,name)
    return satellites
import math
# 计算损耗，d单位为千米，f单位为MHz
# 已知发射器的频率为14.5GHz
def fspl(d,f):
    return 20 * math.log10(d) + 20 * math.log10(f) + 32.44

def distance_point_to_segment(point, segment):
    """
    点到线段的距离
    :param point:
    :param segment:
    :return:
    """
    # 线段端点
    p1, p2 = segment
    # 线段方向向量
    v = p2 - p1
    # 点到p1的向量
    v1 = point - p1
    # 点到直线距离
    distance_to_line = np.linalg.norm(np.cross(v1, v)) / np.linalg.norm(v)
    # 计算投影点
    projection = p1 + np.dot(v1, v) / np.dot(v, v) * v
    # 判断投影点是否在线段上
    if np.dot(projection - p1, projection - p2) <= 0:
        return distance_to_line
    else:
        # 点到线段两个端点的距离
        distance_to_endpoints = [np.linalg.norm(point - p1), np.linalg.norm(point - p2)]
        return min(distance_to_line, *distance_to_endpoints)


def communicable(s:Satellite,g:GroundStation,time):
    """
    计算查看是否可以通信
    :param s:
    :param g:
    :param time:
    :return:
    """
    s_at=s.get_position_at_time(time)
    g_at=g.get_position_at_time(time)
    # print(s_at,g_at)
    point=np.array([0,0,0])
    segment=[np.array(s_at),np.array(g_at)]
    range=distance_point_to_segment(point=point,segment=segment)
    return range
    
class Figure:

    def __init__(self,time=datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x, y, z, color='b', alpha=0.1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.default_time=time
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Satellite Orbits {time}')

    def add_line(self, positions, name):
        x = []
        y = []
        z = []
        for position in positions:
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])
        line1, = self.ax.plot(x, y, z)
        line1.set_label(name)

    def add_point(self, position, name=None):
        if name == None:
            self.ax.scatter(position[0], position[1], position[2])
        else:
            self.ax.scatter(position[0], position[1], position[2], label=name)

    def add_satellite(self, satellite: Satellite):
        self.add_line(satellite.get_path(), satellite.name)
        self.add_point(satellite.get_position_at_time(ts.utc(self.default_time)))  # 使用全局时间变量
        # print(f"{satellite.get_position_at_time(ts.utc(self.default_time))} time={self.default_time}")

    def add_ground_station(self, station: GroundStation):
        self.add_point(station.get_position_at_time(self.default_time), name=station.name)
        print(station.get_position_at_time(self.default_time))

    def add_text(self, position, text):
        self.ax.text(position[0], position[1], position[2], text)
    
    def show(self):
        self.ax.legend()
        plt.show()
    
    def update_time(self,d):
        self.default_time=d


if __name__ == "__main__":
    tle_path="D:\code\stk\starlink.txt"
    sates=read_tle_file(tle_path)
    satellites=[
        sates["BLUEWALKER 3"],
        sates["ION SCV-009"],
        sates["SHERPA-LTC2"],
        sates["STARLINK-1032"],
        sates["STARLINK-1041"],
        sates["STARLINK-1130 (DARKSAT)"],
        sates["STARLINK-1136"],
        sates["STARLINK-1155"],
        sates["STARLINK-1172"]
        ]
    ground_station = GroundStation(latitude_degrees=34.27, longitude_degrees=108.94, name="xian")
    
    start_time=datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)
    end_time=start_time+timedelta(days=1)
    while start_time<=end_time:
        start_time+=timedelta(minutes=1)
        list=[]
        able=[]
        for i in satellites:
            tmp=communicable(i,ground_station,start_time)
            list.append(tmp)
            
            if tmp>=6371:
                able.append(i)
        
        for i in list:
            if i>=6371:
                ftmp=Figure(start_time)
                # print(f"start={start_time}")
                # print(f"ftmp.default_time={ftmp.default_time}")
                for satellite in satellites: 
                    ftmp.add_satellite(satellite)
                ftmp.add_ground_station(ground_station)
                tmp="avalable satellites:\n"
                for s in able:
                    tmp=tmp+s.name+"\n"
                ftmp.add_text([0,0,0],tmp)
                # print()
                ftmp.show()
                break 