import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from etc.flowunit import *
from multiprocessing import Process, Queue, Manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_path', type=str, nargs='?', default='/home/orangepi/workspace/modelbox/graph/person_car.json')
    args = parser.parse_args()

    # 初始化数据
    data = {"frame": None}
    config = {}

    # 读取流程图
    with open(args.graph_path) as f:
        graph = json.load(f)

    # 创建队列
    queue_dict = {}
    queue_size = graph["queue_size"]
    for queue_name in graph["queue_list"]:
        queue_dict[queue_name] = Queue(maxsize=queue_size)

    with Manager() as manager: 
        # 创建共享字典
        share_dict = manager.dict()

        # 创建进程
        process_list = []
        graph_edge = graph["graph_edge"]
        for process in graph_edge.keys():
            p = Process(target=eval(list(graph_edge[process].keys())[0]), args=(share_dict, list(graph_edge[process].values())[0], queue_dict, data,))
            process_list.append(p)

        print("=============Start Process...=============")

        # 启动进程
        for p in process_list:
            p.start()

        # 等待进程结束
        for p in process_list:
            p.join()

        print("==========All Process Finished.===========")