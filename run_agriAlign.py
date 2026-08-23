import os
import argparse
import logging
import sys
import json
from datetime import datetime

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from models.agri_align_model import AgriAlignNERModel
from modules.agri_align_trainer import AgriAlignNERTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = {
    'Disease7000-Refined': {
        'train': 'data/Disease7000-Refined/train.txt',
        'dev': 'data/Disease7000-Refined/valid.txt',
        'test': 'data/Disease7000-Refined/test.txt',
        'train_auximgs': 'data/Disease7000-Refined/Disease7000-Refined_train_dict.pth',
        'dev_auximgs': 'data/Disease7000-Refined/Disease7000-Refined_val_dict.pth',
        'test_auximgs': 'data/Disease7000-Refined/Disease7000-Refined_test_dict.pth'
    },
}

IMG_PATH = {
    'Disease7000-Refined': 'data/Disease7000-Refined_images',
}

AUX_PATH = {
    'Disease7000-Refined': {
        'train': 'data/Disease7000-Refined_aux_images/train/crops',
        'dev': 'data/Disease7000-Refined_aux_images/val/crops',
        'test': 'data/Disease7000-Refined_aux_images/test/crops',
    }
}

class MMPNERProcessor():
    '''
    This is Data Processor...

    '''
    ...

class MMPNERDataset(Dataset):
    ...

def set_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def run_experiment(args):
    data_path = DATA_PATH[args.dataset_name]
    img_path = IMG_PATH[args.dataset_name]
    aux_path = AUX_PATH[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    set_seed(args.seed)
    if args.save_path is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ablation_name = args.ablation if args.ablation else 'full'
        args.save_path = os.path.join(args.save_path,
                                      f"{args.dataset_name}_{ablation_name}_{timestamp}")
        os.makedirs(args.save_path, exist_ok=True)

    logger.info(f"Save path: {args.save_path}")

    if not args.use_prompt:
        img_path, aux_path = None, None
    processor = MMPNERProcessor(data_path, args.bert_name)

    train_dataset = MMPNERDataset(processor, transform, img_path, aux_path,
                                  args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, pin_memory=False)

    dev_dataset = MMPNERDataset(processor, transform, img_path, aux_path,
                                args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=2, pin_memory=False)

    test_dataset = MMPNERDataset(processor, transform, img_path, aux_path,
                                 args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2, pin_memory=False)
    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())

    model = AgriAlignNERModel(label_list, args, use_amgca=args.use_amgca)
    trainer = AgriAlignNERTrainer(
        train_data=train_dataloader,
        dev_data=dev_dataloader,
        test_data=test_dataloader,
        model=model,
        label_map=label_mapping,
        args=args,
        logger=logger
    )
    results = {}
    if args.do_train:
        trainer.train()
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        results['train'] = {
            'best_dev_f1': trainer.best_dev_metric,
            'best_dev_epoch': trainer.best_dev_epoch
        }
    if args.do_train or args.only_test:
        test_results = trainer.test()
        results['test'] = test_results
    torch.cuda.empty_cache()
    return results
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='Disease7000-Refined', type=str,
                        help="Dataset name: Disease7000-Refined")
    parser.add_argument('--bert_name', default='', type=str,
                        help="Pretrained BERT model")
    parser.add_argument('--max_seq', default=256, type=int,
                        help="Maximum sequence length")
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=3e-6, type=float)
    parser.add_argument('--warmup_ratio', default=0.05, type=float)
    parser.add_argument('--eval_begin_epoch', default=5, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of gradient accumulation steps (use larger value to reduce memory)")
    parser.add_argument('--fp16', action='store_true',
                        help="Use mixed precision training to reduce memory")
    parser.add_argument('--prompt_len', default=10, type=int)
    parser.add_argument('--prompt_dim', default=800, type=int)
    parser.add_argument('--use_prompt', action='store_true',
                        help="Use visual prompt (multimodal)")
    parser.add_argument('--use_amgca', action='store_true',
                        help="Use AMGCA innovation module")
    parser.add_argument('--save_path', default='checkpoints', type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--ignore_idx', default=0, type=int)

    args = parser.parse_args()

    logger.info(f"Arguments: {args}")



if __name__ == "__main__":
    main()
