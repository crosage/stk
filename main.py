import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import Topos, load, EarthSatellite, utc
from datetime import datetime


class Satellite:

    def __init__(self, line1, line2, name):
        self.satellite = EarthSatellite(line1, line2, name)
        self.ts = load.timescale()
        self.name=name
        self.path = []

    def get_position_at_time(self, time):
        return self.satellite.at(time).position.km

    def get_position_over_time(self, start_time, end_time, interval_seconds=60):
        times = []
        start_time = self.ts.utc(start_time)
        end_time = self.ts.utc(end_time)
        while start_time < end_time:
            times.append(start_time)
            start_time = start_time + interval_seconds / 86400
        # times=self.ts.utc(start_time,end_time,interval_seconds=interval_seconds)
        # print(times)
        positions = [self.get_position_at_time(t) for t in times]
        return positions

    def get_path(self):
        start_time = datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc)
        end_time = datetime(2024, 2, 6, 14, 0, 0, tzinfo=utc)
        positions = self.get_position_over_time(start_time, end_time)
        return positions

def distance_satellite_to_satellite(s1:Satellite,s2:Satellite):
    ss=s1.get_position_at_time(datetime(2024, 2, 6, 12, 0, 0, tzinfo=utc))
    print(ss)
def distance_point_to_segment(point,segment):
    """
    点到线段的距离
    :param point:
    :param segment:
    :return:
    """
    #线段端点
    p1,p2=segment
    #线段方向向量
    v=p2-p1
    #点到p1的向量
    v1=point-p1
    #点到直线距离
    distance_to_line=np.linalg.norm(np.cross(v1,v))/ np.linalg.norm(v)
    #计算投影点
    projection=p1+np.dot(v1,v)/np.dot(v,v)*v
    #判断投影点是否在线段上
    if np.dot(projection-p1,projection-p2)<=0:
        return distance_to_line
    else:
        # 点到线段两个端点的距离
        distance_to_endpoints = [np.linalg.norm(point - p1), np.linalg.norm(point - p2)]
        return min(distance_to_line, *distance_to_endpoints)
class Figure:
    def __init__(self):
        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(111,projection="3d")
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

    def add_line(self,positions,name):
        x=[]
        y=[]
        z=[]
        for position in positions:
            x.append(position[0])
            y.append(position[1])
            z.append(position[2])
        line1, = self.ax.plot(x, y, z)
        line1.set_label(name)
    def show(self):
        self.ax.legend()
        plt.show()
if __name__=="__main__":
    t1 = Satellite('1 44713U 19074A   23288.88062483  .00011329  00000+0  77841-3 0  9997',
                   '2 44713  53.0530 158.2800 0001466  90.7803 269.3354 15.06391354216744',
                   'STARLINK-1007')
    f=Figure()
    f.add_line(t1.get_path(),t1.name)
    f.show()
    t2=Satellite
    # point = np.array([0, 0, 0])
    # segment = [np.array([0, 0, 1]), np.array([1, 1, 0])]
    # print("点到线段的最小距离:", distance_point_to_segment(point, segment))
