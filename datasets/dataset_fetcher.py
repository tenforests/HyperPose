from datasets.images_dataset import ImagesDataset
from datasets.latents_images_dataset import LatentsImagesDataset
from datasets.yuchen_dataset import MyDataset

class DatasetFetcher:

    def get_dataset(self, opts, dataset_args, transforms_dict):
        if opts.dataset_type in ['yc_hypernet']:
            return self.__get_yc_dataset(opts, dataset_args, transforms_dict)
        
        if opts.dataset_type in ['ffhq_hypernet_pre_extract']:
            return self.__get_latents_dataset(opts, dataset_args, transforms_dict)
        else:
            return self.__get_images_dataset(opts, dataset_args, transforms_dict)

    @staticmethod
    def __get_latents_dataset(opts, dataset_args, transforms_dict):
        train_dataset = LatentsImagesDataset(source_root=dataset_args['train_source_root'],
                                             target_root=dataset_args['train_target_root'],
                                             latents_path=dataset_args['train_latents_path'],
                                             source_transform=transforms_dict['transform_source'],
                                             target_transform=transforms_dict['transform_gt_train'],
                                             opts=opts)
        test_dataset = LatentsImagesDataset(source_root=dataset_args['test_source_root'],
                                            target_root=dataset_args['test_target_root'],
                                            latents_path=dataset_args['test_latents_path'],
                                            source_transform=transforms_dict['transform_source'],
                                            target_transform=transforms_dict['transform_test'],
                                            opts=opts)
        return train_dataset, test_dataset

    @staticmethod
    def __get_images_dataset(opts, dataset_args, transforms_dict):
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=opts)
        return train_dataset, test_dataset

    @staticmethod
    def __get_yc_dataset(opts, dataset_args, transforms_dict):
        train_dataset = MyDataset(pose_root=dataset_args['train_source_root'],
                                    gt_root=dataset_args['train_target_root'],
                                    ld_root=dataset_args['train_ld_root'],
                                    index_range=[0,12000],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_gt_train'],
                                    pose_transform=transforms_dict['transform_pose'],
                                    opts=opts)
        test_dataset = MyDataset(pose_root=dataset_args['test_source_root'],
                                     gt_root=dataset_args['test_target_root'],
                                     ld_root=dataset_args['test_ld_root'],
                                     index_range=[12001,14949],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     pose_transform=transforms_dict['transform_source'],
                                     opts=opts)
        return train_dataset, test_dataset