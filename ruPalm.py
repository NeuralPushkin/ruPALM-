import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader



import os

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import  GPT2TokenizerFast

BASE_MODEL = "sberbank-ai/rugpt3small_based_on_gpt2"

tokenizer = GPT2TokenizerFast.from_pretrained(BASE_MODEL, bos_token='<|endoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')




class Args():
      def __init__(self):
        self.train_batch_size = 1 
        self.lr = 1e-4
        self.n_training_steps = 10

args = Args()


class SumDataset(Dataset):
    def __init__(self):
      self.max_length = 1024
      self.prepared = []
      df = pd.read_json('gazeta_train.jsonl',lines=True)
      
      for sum, text in zip(df['summary'],df['text']):
        pr = f'<s>{text} короче говоря {sum}'
        
        enc = self._encode(text=pr)
        if len(enc)<1024:
          self.prepared+=[enc]
      

    def __len__(self):
        return len(self.prepared)

    def __getitem__(self, item):
        return torch.tensor(self.prepared[item], dtype=torch.long)

    def _encode(self, text):
        
        encoded_sample = tokenizer.encode(text, truncation=True, max_length=self.max_length, padding="max_length")

        return encoded_sample

from tqdm import tqdm

class Dataset_processing(Dataset):
    def __init__(self):
        super().__init__()

        

        self.exmpls = []
        self.eos = tokenizer.eos_token
        
        
        
        
        with open('wiki_prepared.txt') as f:
          lines = f.readlines()
        data = []
        for mes in lines[:100000]:
            data+=[mes]
        
        descriptions = data
        
        
        self.max_length = 1024
        
        for text in tqdm(descriptions,desc='Tokenizing'):
          txt = text + self.eos

          try:
            encoded = tokenizer.encode(txt,  truncation=True, max_length=self.max_length, padding="max_length")
            self.exmpls.append(encoded)
          except Exception as e:
            print(e)
          

          
        
        
    def __len__(self):
        return len(self.exmpls)

    def __getitem__(self, idx):
        return torch.tensor(self.exmpls[idx], dtype=torch.long)


class DecoderDataModule(pl.LightningDataModule):

  def __init__(self, args,data = None):
    super().__init__()
   

  def setup(self, stage=None):
    
    print('preparing data')
    self.wiki = Dataset_processing()
    self.sum = SumDataset()
    self.train_dataset = torch.utils.data.ConcatDataset([self.sum, self.wiki])
    
    args.n_training_steps = len(self.train_dataset)//args.train_batch_size
    args.n_warmup_steps = args.n_training_steps//10

   
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=args.train_batch_size,
      shuffle=True
    )

  '''def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=args.eval_batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.eval_batch_size,
      num_workers=2
    )'''

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

class DecoderTrainer(pl.LightningModule):

  def __init__(self, args):
    super().__init__()
    
    self.gpt = AutoregressiveWrapper(TransformerWrapper(
        num_tokens = 50259,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 12,
            heads = 8,
            
              
        )
    ))
    if not args.n_training_steps:
      self.n_training_steps = 25000#args.n_training_steps
    else:
      self.n_training_steps = args.n_training_steps
      self.n_warmup_steps = 10
    

  def forward(self, input_ids):
    
    print(input_ids)
    loss = self.gpt.forward(input_ids)
    return loss

  def training_step(self, batch, batch_idx):
    
    loss = self.gpt(batch)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss}

  def validation_step(self, batch, batch_idx):
  
    loss = self.gpt(batch)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    
    
    loss = self.gpt(batch)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    
    pass

  def configure_optimizers(self):

    optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer, 
      max_lr=args.lr,
      final_div_factor=500, 
      steps_per_epoch=args.n_training_steps,
      epochs=10
    )

    

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

data_module = DecoderDataModule(args)
model = DecoderTrainer(args=args)


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

early_stopping_callback = EarlyStopping(monitor='train_loss', patience=2)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename =  f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="Decoder")

trainer = pl.Trainer(
  logger=wandb_logger,
  checkpoint_callback=[CheckpointEveryNSteps(save_step_frequency=20000)],
  callbacks=[early_stopping_callback],
  max_epochs=10,
  accelerator='ddp',
  gpus=[0, 1],
  gradient_clip_val=0.5,
  accumulate_grad_batches=7,
  progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)
