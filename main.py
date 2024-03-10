import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import Topos, load, EarthSatellite, utc
from datetime import datetime, timedelta

default_time = datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)
ts = load.timescale()


class GroundStation:

    def __init__(self, latitude_degrees, longitude_degrees, name, elevation_meters=0):
        """

        :param latitude_degrees:
        :param longitude_degrees:
        :param elevation_meters: 海拔高度
        """
        self.location = Topos(latitude_degrees, longitude_degrees, elevation_meters)
        self.name = name
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

    def __init__(self, line1, line2, name):
        self.satellite = EarthSatellite(line1, line2, name)
        self.name = name
        self.path = []

    def get_position_at_time(self, time):
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


def distance_satellite_to_satellite(s1: Satellite, s2: Satellite):
    ss = s1.get_position_at_time(default_time)
    print(ss)


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


class Figure:
    def __init__(self):
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
        self.ax.set_zlabel('Z')
        self.ax.set_title('Satellite Orbits')

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

    def add_satellite(self, satellite: Satellite, time=default_time):

        self.add_line(satellite.get_path(), satellite.name)
        self.add_point(satellite.get_position_at_time(ts.utc(time)))
        print(satellite.get_position_at_time(ts.utc(time)))

    def add_ground_station(self, station: GroundStation, time=default_time):
        self.add_point(station.get_position_at_time(time), name=station.name)

    def show(self):
        self.ax.legend()
        plt.show()


if __name__ == "__main__":
    t1 = Satellite('1 44713U 19074A   23288.88062483  .00011329  00000+0  77841-3 0  9997',
                   '2 44713  53.0530 158.2800 0001466  90.7803 269.3354 15.06391354216744',
                   'STARLINK-1007')
    t2 = Satellite('1 44743U 19074AG  23289.11746495  .00037658  00000+0  25301-2 0  9995',
                   '2 44743  53.0546 197.2150 0001327  88.5553 271.5588 15.06450493216393',
                   'STARLINK-1038')
    ground_station = GroundStation(latitude_degrees=51.5074, longitude_degrees=-0.1278, name="lond")
    f = Figure()
    f.add_satellite(t1)
    f.add_satellite(t2)
    f.add_ground_station(ground_station)
    f.show()
    # t2=Satellite
    # point = np.array([0, 0, 0])
    # segment = [np.array([0, 0, 1]), np.array([1, 1, 0])]
    # print("点到线段的最小距离:", distance_point_to_segment(point, segment))
