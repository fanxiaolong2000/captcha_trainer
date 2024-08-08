from make_dataset import DataSets
from config import ModelConfig
from trains import Trains
import json
from constants import DatasetType,RunMode
import os
# 打包
def add_data():
    train_project = "./resource/train_set2"
    model =  ModelConfig('test-CNNX-GRU-H64-CTC-C1')
    ds = DataSets(model)
    ds.make_dataset(train_project,train_project,True,msg=print)

    # attached_dataset_name = "dataset/{}".format(model.dataset_increasing_name(RunMode.Trains))
    # attached_dataset_path = os.path.join(fr"./projects/{train_project}", attached_dataset_name).replace("\\", '/')
    attached_dataset_name = "dataset/{}".format(model.dataset_increasing_name(RunMode.Validation))
    attached_dataset_path = os.path.join(fr"./projects/{train_project}", attached_dataset_name).replace("\\", '/')
    model.trains_path[DatasetType.TFRecords].append(attached_dataset_path)
    model.trains_path[DatasetType.Directory].append(train_project)
    # model.validation_path[DatasetType.TFRecords].append()
    # model.validation_path[DatasetType.Directory].append()
    model.update()
    
    

def training_task(project_name):
        model_conf = ModelConfig(project_name)

        current_task = Trains(model_conf)
        try:
            current_task.train_process()
            status = '训练完成'
        except Exception as e:
            
            print(
                e.__class__.__name__, json.dumps(e.args, ensure_ascii=False)
            )
if __name__ == '__main__':
     add_data()
     # training_task('test-CNNX-GRU-H64-CTC-C1')