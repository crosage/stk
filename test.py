import numpy as np

# 定义起点和终点
i = np.array([-816.07604579, 5515.67266438, 4071.52501339])
g = np.array([-2917.03462092, 4391.38042962, 3577.89875371])

# 生成参数 t 从 0 到 1 的线性空间
num_points = 100  # 你可以根据需要调整生成点的数量
t = np.linspace(0, 1, num_points)

# 计算连线上所有点的坐标
points = np.outer(1 - t, i) + np.outer(t, g)
pointss=np.linspace(i,g,100)
for i in range(len(points)):
    print(points[i],pointss[i])
# print(points)

# 计算每对相邻点之间的向量
vectors = points[1:] - points[:-1]

# 计算第一个向量的方向
direction = vectors[0] / np.linalg.norm(vectors[0])

# 验证所有向量是否与第一个向量平行
for v in vectors:
    v_norm = v / np.linalg.norm(v)
    if not np.allclose(v_norm, direction):
        print("点不在一条直线上")
        break
else:
    print("所有点都在一条直线上")
