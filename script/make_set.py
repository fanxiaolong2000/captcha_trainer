from make_dataset import DataSets
train_path = "../resource/train_set"
ds = DataSets('test-CNNX-GRU-H64-CTC-C1')
ds.make_dataset(train_path,train_path,True,msg=print)