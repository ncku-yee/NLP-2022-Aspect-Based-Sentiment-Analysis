# import os

# pretrained_list = [
#     # 'fnlp/bart-base-chinese',
#     # 'fnlp/bart-large-chinese',
#     # 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI',
#     # 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment',
#     # 'IDEA-CCNL/Erlangshen-Roberta-110M-Similarity',
#     # 'hfl/chinese-macbert-large',
#     # 'hfl/chinese-roberta-wwm-ext-large',
#     'symanto/xlm-roberta-base-snli-mnli-anli-xnli',
#     'nghuyong/ernie-gram-zh',
# ]

# for pretrained_model in pretrained_list:
#     if 'bart-large' in pretrained_model.lower():
#         batch_size, accum_steps = 4, 16
#     elif 'large' in pretrained_model.lower():
#         batch_size, accum_steps = 8, 8
#     else:
#         batch_size, accum_steps = 16, 8
#     os.system(f"python train.py --task 2 --batch_size {batch_size} --accum_steps {accum_steps} --pretrained_model {pretrained_model} --drop_prob {0.3} --num_epochs {10}")


import os

# Root CONFIG directory
config_dir = "./configs"
assert os.path.exists(config_dir), f"{config_dir} is not found"

# Each task directory
task_dir = sorted([os.path.join(config_dir, x) for x in os.listdir(config_dir)])

# Each task's config files
task_configs = {}

# Get all config files
for dir in task_dir:
    config_files = sorted([os.path.join(dir, x) for x in os.listdir(dir)])
    assert config_files, f"No CONFIG files in {dir}"
    task_configs[dir] = config_files

# # Run the train.py with given configuration YAML
# for dir, files in task_configs.items():
#     for file in files:
#         os.system(f"python train.py --config {file}")

for file in task_configs[task_dir[0]]:
    os.system(f"python train.py --config {file}")