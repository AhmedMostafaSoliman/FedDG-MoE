import os
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from configs.default import vlcs_path
from torchvision import transforms
import torch

transform_train = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

vlcs_name_dict = {
    'v': 'PASCAL',
    'l': 'LABELME',
    'c': 'CALTECH',
    's': 'SUN',
}

split_dict = {
    'train': 'train',
    'val': 'crossval',
    'test': 'test',
}


class VLCS_SingleDomain():
    def __init__(self, root_path=vlcs_path, domain_name='v', split='test', train_transform=None):
        if domain_name in vlcs_name_dict.keys():
            self.domain_name = vlcs_name_dict[domain_name]
            self.domain_label = list(vlcs_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in v l c s')
        
        self.root_path = root_path
        self.split = split
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
                
        imgs, labels = VLCS_SingleDomain.get_domain_data(self.split, self.domain_name, self.root_path)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)
    
    @staticmethod
    def get_domain_data(split, domain_name, root_path):
        images = []
        labels = []
        raw_images_path = os.path.join(root_path, 'raw_images', domain_name, split_dict[split])
        for i, cls in enumerate(os.listdir(raw_images_path)):
            for img in sorted(os.listdir(os.path.join(raw_images_path, cls))):
                img_path = os.path.join(raw_images_path, cls, img)
                images.append(img_path)
                labels.append(i)
        return images, labels

    
class VLCS_FedDG():
    def __init__(self, test_domain='v', batch_size=16, test_batch_size=32):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.domain_list = list(vlcs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        print(self.train_domain_list)
        print(self.test_domain)
        self.train_domain_list.remove(self.test_domain)  
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = VLCS_FedDG.SingleSite(
                domain_name, 
                self.batch_size, 
                self.test_batch_size if domain_name == self.test_domain else self.batch_size
            )
            
        
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(domain_name, batch_size=16, test_batch_size=None):
        if test_batch_size is None:
            test_batch_size = batch_size

        dataset_dict = {
            'train': VLCS_SingleDomain(domain_name=domain_name, split='train', train_transform=transform_train).dataset,
            'val': VLCS_SingleDomain(domain_name=domain_name, split='val').dataset,
            'test': VLCS_SingleDomain(domain_name=domain_name, split='test').dataset,
        }
        dataloader_dict = {
            'train': torch.utils.data.DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(dataset_dict['val'], batch_size=batch_size, shuffle=False),
            'test': torch.utils.data.DataLoader(dataset_dict['test'], batch_size=test_batch_size, shuffle=False)
        }
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict