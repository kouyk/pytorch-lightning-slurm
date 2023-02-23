import torch
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("seed_everything", "data.init_args.seed")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = CustomCLI(save_config_callback=None)
