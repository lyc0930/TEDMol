import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model.MoLa import MoLa
from data_provider.pretrain_datamodule import GINPretrainDataModule


def main(args):
    pl.seed_everything(args.seed)

    # model
    model = MoLa(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        bert_hidden_dim=args.bert_hidden_dim,
        pretrain=args.pretrain,
        projection_dim=args.projection_dim,
        weight_decay=args.weight_decay,
        init_lr=args.init_lr,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_steps=args.warmup_steps,
        lr_decay_rate=args.lr_decay_rate,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = GINPretrainDataModule.from_argparse_args(args)
    dm.train_dataset.tokenizer = model.text_encoder.tokenizer

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="checkpoints/"+args.filename+"/",
                                         filename='{epoch:02d}',
                                         every_n_epochs=10,
                                         save_top_k=-1))
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         # accumulate_grad_batches=8,
                                         )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="MoLa_v0")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser = Trainer.add_argparse_args(parser)
    parser = MoLa.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args

    parser.set_defaults(batch_size=64,
                        accelerator='gpu',
                        gpus='0,1,2,3,4,5,6,7',
                        # precision=16,
                        max_epochs=200,
                        num_workers=8,
                        root='dataset/MoLa-D',
                        warmup_steps=200
                        )
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")

    main(args)
