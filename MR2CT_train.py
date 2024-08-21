from share import *
from datetime import datetime
import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MR2CTforLungSeg.MR2CT_dataset_elasticdeform import StudyDataset
from cldm.logger_medical import ImageLogger
from cldm.logger_val_metric import LoggerValMetric
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configs
model_config = config['model_config']
trainer_config = config['trainer_config']
dataset_config = config['dataset_config']
logger_config = config['logger_config']

# Model setup
model = create_model(model_config['model_path']).cpu()
model.load_state_dict(load_state_dict(model_config['resume_path'], location='cpu'))
model.learning_rate = model_config['learning_rate']
model.sd_locked = model_config['sd_locked']
model.only_mid_control = model_config['only_mid_control']

# Checkpoints setup
checkpoint = ModelCheckpoint(
    save_top_k=2,
    every_n_epochs=1,
    save_on_train_epoch_end=True,
    save_last=True,
    filename='trloss-{epoch}-{step}',
    monitor="train/loss",
)
checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

checkpoint_val = ModelCheckpoint(
    save_top_k=2,
    every_n_epochs=1,
    save_on_train_epoch_end=False,
    save_last=False,
    filename='val-{epoch}-{step}',
    monitor="val/metric",
    mode='max'
)
checkpoint_val.CHECKPOINT_NAME_LAST = "{epoch}-{step}-last"

# Dataset and DataLoader setup
train_dataset = StudyDataset(type='training', json_path=dataset_config['json_path'], deform=dataset_config['deform'], ch=dataset_config['ch'])
val_dataset = StudyDataset(type='validation', json_path=dataset_config['json_path'], ch=dataset_config['ch'])
train_dataloader = DataLoader(train_dataset, num_workers=30, batch_size=trainer_config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=30, batch_size=trainer_config['batch_size']*2, shuffle=False)

# Logger setup
split = datetime.today().strftime("%y%m%d") + logger_config['split_postfix']
logger = ImageLogger(batch_frequency=logger_config['batch_frequency'], max_images=logger_config['max_images'], split=split)
logger_val_metric = LoggerValMetric(batch_frequency=1, split=split)

# Trainer setup
trainer = pl.Trainer(
    gpus=trainer_config['gpus'],
    precision=trainer_config['precision'],
    callbacks=[logger, logger_val_metric, checkpoint, checkpoint_val],
    max_epochs=trainer_config['max_epochs'],
    check_val_every_n_epoch=trainer_config['check_val_every_n_epoch']
)

# Train!
trainer.fit(model, train_dataloader, val_dataloader)