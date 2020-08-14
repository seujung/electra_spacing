import os
from argparse import Namespace
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from electra_spacing.train_lightning import KoELECTRASpacing

lr_logger = LearningRateLogger()

parser = {}
parser['gpus'] = 1

hparams = {}
hparams['file_path'] = './input_data.tsv'
hparams["max_epochs"] = 50
hparams['train_ratio'] = 0.7
hparams['batch_size'] = 64
hparams['weight_value'] = 10
hparams['learning_rate'] = 0.001
hparams['warmup_ratio'] = 0.2

parser = Namespace(**parser)
hparams = Namespace(**hparams)
callbacks=[lr_logger],
trainer = Trainer.from_argparse_args(parser, callbacks=[lr_logger] )
# trainer = Trainer(max_epochs = parser["max_epochs"], gpu=parser['gpus'])

model = KoELECTRASpacing(hparams)
trainer.fit(model)