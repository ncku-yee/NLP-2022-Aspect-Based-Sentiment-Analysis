import argparse
import os
import yaml
import torch
from Network import MODELS


class BaseConfig():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        """ Specify the device """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        parser.add_argument('--device', type=str, default=device, required=False, help='Specify the device')
        self.parser = self.InitializeParser(parser)

    def InitializeParser(self, parser):
        """ Initialize the common configuration. """

        parser.add_argument('--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')

        """ Whether to enable fp16 acceleration """
        parser.add_argument('--fp16_training', action="store_false", help='Whether to enable fp16 acceleration(default: True)')

        """ Specify the task """
        parser.add_argument('--task', type=int, default=1, help='Task identifier(default: 1)')

        """ Training hyperparameters """
        parser.add_argument('--seed', type=int, default=0, help='Random seed(default: 0)')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size(default: 8)')
        parser.add_argument('--num_workers', type=int, default=4, help='Batch size(default: 4)')
        parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs(default: 10)')
        parser.add_argument('--accum_steps', type=int, default=8, help='Gradient accumulation steps(default: 8)')
        parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate(default: 5e-5)')

        """ Training information """
        parser.add_argument('--is_train', action="store_false", default=True, help='Whether to train(default: True')
        parser.add_argument('--verbose', action="store_false", default=True, help='Whether to enable logging(default: True)')
        parser.add_argument('--logging_step', type=int, default=100, help='Logging frequency(default: 100)')
        parser.add_argument('--validation', action="store_false", default=True, help='Whether to enable validation(default: True)')

        """ Pretrained model and tokenizer """
        parser.add_argument('--drop_prob', type=float, default=0.3, help='Dropout probability(default: 0.3)')
        parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length(default: 512)')
        parser.add_argument('--pretrained_model', type=str, help='Pretrained model name')

        """ Backup information """
        parser.add_argument('--model_path', type=str, default='', help='Model save path')
        parser.add_argument('--models_dir', type=str, default='./models', help='Models directory(default: ./models)')
        parser.add_argument('--output_path', type=str, default='', help='Output path')

        return parser

    def parse_args(self):
        return self.parser.parse_args()


class TrainConfig(BaseConfig):

    def parse_args(self):
        args = self.parser.parse_args()

        """ Given Specific YAML configuration """
        if args.config:
            assert os.path.exists(args.config), f"{args.config} is not found"
            config = read_yaml(args.config)
            self.parser.set_defaults(**config)
            args = self.parser.parse_args()

        # Check the pretrained model name.
        assert args.pretrained_model in MODELS, f"{args.pretrained_model} is not available"
        # Check the task identifier
        assert args.task in [1, 2], f"Task {args.task} is not supported"

        """ Model save path """
        # No model path specified.
        if not args.model_path:
            model_path = f"best_model_task{args.task}_{args.pretrained_model.split('/')[-1].replace('-', '_').lower()}.pt"
        else:
            model_path = args.model_path
        model_path = os.path.join(args.models_dir, model_path)
        args.model_path = model_path

        """ Output path """
        if not args.output_path:
            output_path = f"prediction_task{args.task}_{args.pretrained_model.split('/')[-1].replace('-', '_').lower()}.csv"
        else:
            output_path = args.output_path
        args.output_path = output_path

        # Visualize the configuration.
        for k, v in sorted(vars(args).items()):
            default = self.parser.get_default(k)        # Default value
            print(f"{k.ljust(20, ' ')}: {v}")
        return args


def read_yaml(file):
    assert os.path.exists(file), f"{file} is not found"
    assert file.endswith('.yml'), "Not YAML file"
    with open(file, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config

if __name__ == "__main__":
    parser = TrainConfig()
    args = parser.parse_args()