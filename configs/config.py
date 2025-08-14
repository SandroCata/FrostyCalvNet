from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "AMD_HookNet"

__C.CUDA = True

__C.output_path = './checkpoints'
__C.output_path_predictions = './predictions'


# model
__C.num_workers=16
__C.in_channels = 1
__C.n_classes = 2

__C.n_filters = 8               
__C.filter_size = 3              
__C.learning_rate = 0.001        
__C.gamma=0.99
__C.total_epoch = 5       
__C.scale = 1.0                    
__C.batch_size = 16


__C.load = True
__C.seed = 123            



