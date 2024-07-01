import os

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
from options.test_options import TestOptions
from utils import util
from collections import OrderedDict
def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results_5w')
    # out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    # os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    
    dataset = InferenceDataset(pose_root=dataset_args['test_source_root'],
                                     gt_root=dataset_args['test_target_root'],
                                     ld_root=dataset_args['test_ld_root'],
                                     index_range=[12001,14949],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     pose_transform=transforms_dict['transform_source'],
                                     opts=opts,
                                     random=False)
    # dataset = InferenceDataset(root=opts.data_path,
    #                            transform=transforms_dict['transform_inference'],
    #                            opts=opts)
    
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    if "cars" in opts.dataset_type:
        resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
    else:
        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    all_latents = {}
    for (ref_img,gt_img,pose_img,org_pose) in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            ref_img,gt_img,pose_img,org_pose =ref_img.cuda().float(),gt_img.cuda().float(),pose_img.cuda().float(),org_pose.cuda().float()
            # input_cuda = input_batch.cuda().float()
            tic = time.time()
            result,_,_,_ = run_inversion(ref_img,pose_img, net, opts, return_intermediate_results=False)
            toc = time.time()
            print(toc-tic)
            global_time.append(toc - tic)
        total_distance, total_pixels = 0, 0
        mtotal_distance, mtotal_pixels = 0, 0
        # mouth_total_distance, mouth_total_pixels = 0, 0
        # for i in range(ref_img.shape[0]):
        
        fake_frame = util.tensor2im(result)
        rgb_frame = util.tensor2im(gt_img)
        nmfc_frame = util.tensor2im(org_pose)
        visual_list = [('real', rgb_frame),
                    ('fake', fake_frame),
                    ('nmfc', nmfc_frame)]
        # for i in range(fake_frame.shape[0]):
        # fake_frame = np.array(fake_frame)
        # rgb_frame = np.array(rgb_frame)
        # nmfc_frame = np.array(nmfc_frame)
        total_distance, total_pixels, heatmap = util.get_pixel_distance(
                    rgb_frame, fake_frame, total_distance, total_pixels)
        mtotal_distance, mtotal_pixels, mheatmap = util.get_pixel_distance(
                rgb_frame, fake_frame, mtotal_distance, mtotal_pixels, nmfc_frame)
        # heatmap = heatmap.tolist()
        # mheatmap = mheatmap.tolist()
        visual_list += [('heatmap', heatmap),
                        ('masked_heatmap', mheatmap)]
        visuals = OrderedDict(visual_list)
        for label, image_numpy in visuals.items():
            util.mkdir(os.path.join(out_path_results, label))
            image_name = f'{global_i:06d}.png'
            save_path = os.path.join(out_path_results, label, image_name)
            util.save_image(image_numpy, save_path)
        global_i += 1
    output_video_cmd = f'python utils/images_to_video.py --results_dir {out_path_results} --output_mode {opts.output_mode}'
    os.system(output_video_cmd)
            # visualizer = Visualizer(opt)
            # visualizer.save_images(save_dir, visuals, img_path[-1])
            
            # results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
            # im_path = dataset.paths[global_i]

            # input_im = tensor2im(input_batch[i])
            # res = np.array(input_im.resize(resize_amount))
            # for idx, result in enumerate(results):
            #     res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            #     # save individual outputs
            #     save_dir = os.path.join(out_path_results, str(idx))
            #     os.makedirs(save_dir, exist_ok=True)
            #     result.resize(resize_amount).save(os.path.join(save_dir, os.path.basename(im_path)))

            # # save coupled image with side-by-side results
            # Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            # all_latents[os.path.basename(im_path)] = result_latents[i][0]

            # if opts.save_weight_deltas:
            #     weight_deltas_dir = os.path.join(test_opts.exp_dir, "weight_deltas")
            #     os.makedirs(weight_deltas_dir, exist_ok=True)
            #     np.save(os.path.join(weight_deltas_dir, os.path.basename(im_path).split('.')[0] + ".npy"),
            #             result_deltas[i][-1])

    apd_res = ' %0.2f' % (total_distance/total_pixels)
    mapd_res = '%0.2f' % (mtotal_distance/mtotal_pixels)
    print('Average pixel (L2) distance for sequence (APD-L2): %0.2f' % (total_distance/total_pixels))
            # Masked Average Pixel Distance (MAPD-L2)
    print('Masked average pixel (L2) distance for sequence (MAPD-L2): %0.2f' % (mtotal_distance/mtotal_pixels))
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)
        f.write(f'Average pixel (L2) distance for sequence (APD-L2): {apd_res}\n')
        f.write(f'Masked average pixel (L2) distance for sequence (MAPD-L2): {mapd_res}\n')
    # save all latents as npy file
    # np.save(os.path.join(test_opts.exp_dir, 'latents.npy'), all_latents)


if __name__ == '__main__':
    run()
