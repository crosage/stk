import pandas as pd
import math
import json
def deal_C(Access_path,Bent_Pipe_path,name,output_path):
    access=pd.read_csv(Access_path)
    bentpipe=pd.read_csv(Bent_Pipe_path)
    # if len(access) != len(bentpipe):
        # raise ValueError("两个表格的行数不同，无法进行对应行的计算")
    index=0
    data_list=[]
    for (access_index, access_data), (bentpipe_index, bentpipe_data) in zip(access.iterrows(), bentpipe.iterrows()):
        time1=access_data["Time (UTCG)"]
        time2=bentpipe_data["Time (UTCG)"]
        if time1!=time2:
            print(f"{time1} {time2} {name}")
            raise ValueError("两个表格的行数不同，无法进行对应行的计算")
        # 距离
        range=access_data["Range (km)"]
        # 信噪比
        SNR=bentpipe_data["C/N1 (dB)"]
        # 带宽
        bandwidth=bentpipe_data["Bandwidth1 (kHz)"]
        # 时延
        t=range*2*1000/300000000
        # 信道容量
        C=bandwidth*math.log2(1+SNR)
        data = {
            "name": name,
            "time": time1.replace(" ","-"),
            "range": range,
            "SNR": SNR,
            "t": t,
            "channel_capacity": C
        }
        data_list.append(data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    return json.dumps(data_list)

def deal_namae(name):
    path=f"D:\\code\\stk\\xian_to_{name}_Access_AER.csv"
    path2=f"D:\\code\\stk\\xian_to_{name}_Bent_Pipe_Comm_Link.csv"
    output_path=f"D:\\code\\stk\\xian_to_{name}.json"
    deal_C(path,path2,name,output_path)

if __name__ == "__main__":
    # deal_namae("starlink-1231")
    # deal_namae("sherpa-ltc2")
    # deal_namae("starlink-1032")
    # deal_namae("bluewalker_3_53807")
    # deal_namae("ion_scv-009_55441")
    # deal_namae("starlink-1041")
    # deal_namae("starlink-1130_darksat")
    # deal_namae("starlink-1136")
    # deal_namae("starlink-1155")
    deal_namae("starlink-1172")