from share import *
from datetime import datetime

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MR2CTforLungSeg.MR2CT_dataset_elasticdeform import StudyDataset
from cldm.logger_medical import ImageLogger
from cldm.logger_val_metric import LoggerValMetric
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

# Configs
resume_path = './models/control_sd15_ini.ckpt'

max_epochs = 500
batch_size = 4
logger_freq = 2500
logger_freq_epoch = 50
learning_rate = 6e-7 # Adjust experimentally to find the best value for your dataset
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Checkpoint
checkpoint = ModelCheckpoint(
    save_top_k=2,
    # every_n_train_steps=logger_freq,
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    save_last=False,
    filename='trloss-{epoch}-{step}',
    monitor="train/loss",
)
checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

checkpoint_val = ModelCheckpoint(
    save_top_k=2,
    every_n_epochs=1,
    save_on_train_epoch_end=False, # val이면 false, train이면 true
    save_last=False,
    filename='val-{epoch}-{step}',
    monitor="val/metric",
    mode='max'
)
checkpoint_val.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

# checkpoint_epochEnd = ModelCheckpoint(
#     save_top_k=-1,
#     every_n_epochs=logger_freq_epoch,
#     save_on_train_epoch_end=True,
#     save_last=False,
#     filename='end-{epoch}-{step}',
#     monitor=None,
# )
# checkpoint_epochEnd.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"


# Misc
train_dataset = StudyDataset(type='training', json_path='your_filename.json', deform=10, ch=3)
# deform: 
# ch: 3 means use channel diverisy, 1 means use copy channel 1
val_dataset = StudyDataset(type='validation', json_path='your_filename.json', ch=3)
train_dataloader = DataLoader(train_dataset, num_workers=40, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=40, batch_size=batch_size*2, shuffle=False)

# Folder location: Specifies the directory where the logger will save files during model training.
split = datetime.today().strftime("%y%m%d")+"_your_postfix"

logger = ImageLogger(batch_frequency=logger_freq, max_images=4, split=split)
logger_val_metric = LoggerValMetric(batch_frequency=1, split=split)

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, logger_val_metric, checkpoint, checkpoint_val], max_epochs=max_epochs,
                     check_val_every_n_epoch=4)
# precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
# Can be used on CPU, GPU or TPUs.


# Train!
trainer.fit(model, train_dataloader, val_dataloader)