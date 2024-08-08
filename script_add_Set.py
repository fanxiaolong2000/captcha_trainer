from make_dataset import DataSets
from config import ModelConfig
from trains import Trains
import json
# 打包
def package():
    train_path = "./resource/train_set"
    model =  ModelConfig('test-CNNX-GRU-H64-CTC-C1')
    ds = DataSets(model)
    ds.make_dataset(train_path,train_path,True,msg=print)
    filename =train_path
    if not filename:
        return
    model_conf = model

    if not self.check_dataset(model_conf):
        
        return

    self.attach_dataset_val.set(filename)
    self.sample_map[DatasetType.Directory][RunMode.Trains].insert(tk.END, filename)
    self.button_state(self.btn_attach_dataset, tk.DISABLED)

    for mode in [RunMode.Trains, RunMode.Validation]:
        attached_dataset_name = model_conf.dataset_increasing_name(mode)
        attached_dataset_name = "dataset/{}".format(attached_dataset_name)
        attached_dataset_path = os.path.join(self.project_path, attached_dataset_name)
        attached_dataset_path = attached_dataset_path.replace("\\", '/')
        if mode == RunMode.Validation and self.validation_num_val.get() == 0:
            continue
        self.sample_map[DatasetType.TFRecords][mode].insert(tk.END, attached_dataset_path)
    self.save_conf()
    model_conf = ModelConfig(self.current_project)
    self.threading_exec(
        lambda: DataSets(model_conf).make_dataset(
            trains_path=filename,
            is_add=True,
            callback=lambda: self.button_state(self.btn_attach_dataset, tk.NORMAL),
            msg=lambda x: tk.messagebox.showinfo('附加数据状态', x)
        )
    )
    pass

    

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
     training_task('test-CNNX-GRU-H64-CTC-C1')