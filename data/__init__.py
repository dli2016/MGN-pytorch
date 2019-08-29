from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from data.dataset import GeneralDataLoader
from torch.utils.data import dataloader

class Data:

    def __init__(self, args):
        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, \
                mean=[0.0, 0.0, 0.0]))
        
        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            self.trainset = GeneralDataLoader(args, train_transform, \
                args.data_train.lower(), 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                #shuffle=True,
                batch_size=args.batchid * args.batchimage,
                num_workers=args.nThread)
        else:
            self.train_loader = None

        if args.data_test in ['market1501', 'duke', 'cuhk03', 'rap2']:
            self.testset = GeneralDataLoader(args, test_transform, \
                args.data_test, 'test')
            self.queryset= GeneralDataLoader(args, test_transform, \
                args.data_test, 'query')
        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, \
            batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, \
            batch_size=args.batchtest, num_workers=args.nThread)
