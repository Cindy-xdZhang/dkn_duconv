import torch
import os
import argparse
from tkinter import *  
def arg_config():
    project_root_dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default="transformer-master\\models")
    config = parser.parse_args()
    return config
def main():
    config=arg_config()
    # model=load_model()
    # while True:
    #     input_q=input()
        # input_check()
        # print(input_q)
        # reply=model(input_q)
        # print(reply)
            # 导入 Tkinter 库
    root = Tk()                     # 创建窗口对象的背景色
    # 创建两个列表
    li     = ['C','python','php','html','SQL','java']
    movie  = ['CSS','jQuery','Bootstrap']
    listb  = Listbox(root)          #  创建两个列表组件
    listb2 = Listbox(root)
    for item in li:                 # 第一个小部件插入数据
        listb.insert(0,item)
    for item in movie:              # 第二个小部件插入数据
        listb2.insert(0,item)
    
    listb.pack()                    # 将小部件放置到主窗口中
    listb2.pack()
    root.mainloop()                 # 进入消息循环
        


if __name__ == "__main__":
    main()
