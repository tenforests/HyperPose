import cv2
from PIL import Image
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def write_video_to_file(video_file_path, video, mult_duration=1, n_last_frames_omit=0):
    image_size_h = video.shape[1]
    image_size_w = video.shape[2]
    writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 25 , (image_size_w, image_size_h), True)
    t = 0
    # mult_duration: how slower should the video be - was 3
    while t < video.shape[0] - n_last_frames_omit:
        for _ in range(mult_duration):
            writer.write(video[t,:,:,:])
        t += 1
    writer.release()
    print("Sample saved to: " + video_file_path)

def images_to_video(image_names):
    # Open images
    images = []
    for image_name in tqdm(image_names,desc='images_to_video'):
        images.append(np.array(Image.open(image_name)))
    video = np.array(images)
    return video[:,:,:,::-1]

def make_images_dict(dir):
    images = {}
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))
        if len(paths) > 0:
            folder = os.path.basename(root)
            images[folder] = paths

    return images
def make_images_dict_lcc(dir):
    images = {}
    for root,dir ,file in os.walk(dir):
        paths = []
        for filename in sorted(file):
            if is_image_file(filename):
                paths.append(os.path.join(root, filename))
        if len(paths) > 0:
            folder = os.path.basename(root)
            if folder not in images.keys():
                images[folder] = paths
            else:
                images[folder].extend(paths)
    for key in images.keys():
        images[key] = sorted(images[key])
    return images
def print_args(parser, args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '-------------------------------------------'
    print(message)

def main():
    print('---- Create .mp4 video file from images --- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,default='/home/lcc/results/ccbr_shuwen_20230908_1080x1080_silu/ccbr_shuwen_20230907_1080x1080_silu/latest_epoch/test/',
                        help='Path to the directory where generated images are saved.')
    parser.add_argument('--output_mode', type=str,
                        choices=['only_fake', 'source_target','mask',
                                 'source_nmfc_target', 'heatmap',
                                 'masked_heatmap', 'all_heatmaps', 'all',
                                 'source_target_separate','infer','source_nmfc_target_seperate','my'],
                        default='source_nmfc_target', # 'all'
                        help='What images to save in the video file.')
    parser.add_argument('--save_video_name', type=str,default='reslut',
                        help='Path to the directory where generated images are saved.')
    args = parser.parse_args()
    print_args(parser, args)

    path = args.results_dir
    if path is None or not os.path.isdir(path):
        # os.makedirs(path,exist_ok=True)
        raise ValueError(path + ' path does not exist.')
    else:
        print('Converting images in %s to .mp4 video.' % path)

    image_paths = make_images_dict_lcc(path)
    video_name = args.save_video_name + '.mp4' # path.replace('/', '-') + '.mp4'
    save_path = os.path.join(path, video_name)
    heatmap_video, masked_heatmap_video = None, None
    fake_video = images_to_video(image_paths['fake'])
    print('Processing %d frames...' % fake_video.shape[0])
    nmfc_video = images_to_video(image_paths['nmfc'])
    if args.output_mode == 'mask':
        mask_video = images_to_video(image_paths['mask'])
    if 'eye_gaze' in image_paths:
        eye_gaze_video = images_to_video(image_paths['eye_gaze'])
    else:
        eye_gaze_video = None
    if args.output_mode !='infer':
        rgb_video = images_to_video(image_paths['real'])
        assert rgb_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'
    assert fake_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'
    
    if args.output_mode in ['heatmap', 'all_heatmaps', 'all','masked_heatmap']:
        assert 'heatmap' in image_paths and 'masked_heatmap' in image_paths, 'No heatmap files found.'
        heatmap_video = images_to_video(image_paths['heatmap'])
        masked_heatmap_video = images_to_video(image_paths['masked_heatmap'])
        assert heatmap_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'
        assert masked_heatmap_video.shape[0] == nmfc_video.shape[0], 'Not correct number of image files.'

    if args.output_mode == 'only_fake':
        video_list = [fake_video]
    elif args.output_mode == 'my':
        for i,nmfc in enumerate(nmfc_video):
            rgb_video[i] = 0.5*nmfc+0.5*rgb_video[i]
            fake_video[i] = 0.5*nmfc+0.5*fake_video[i]
        video_list = [rgb_video,fake_video]
    elif args.output_mode == 'source_target' or args.output_mode == 'source_target_separate':
        video_list = [rgb_video, fake_video]
    elif args.output_mode == 'source_nmfc_target' or args.output_mode == 'source_nmfc_target_seperate':
        if eye_gaze_video is not None:
            video_list = [rgb_video, nmfc_video, eye_gaze_video,fake_video]
        else:
            video_list = [rgb_video, nmfc_video, fake_video]
    elif args.output_mode == 'mask':
        video_list = [rgb_video, mask_video, eye_gaze_video, fake_video]
    elif args.output_mode == 'heatmap':
        video_list = [rgb_video, fake_video, heatmap_video]
    elif args.output_mode == 'masked_heatmap':
        video_list = [rgb_video, fake_video, masked_heatmap_video]
    elif args.output_mode == 'all_heatmaps':
        video_list = [rgb_video, fake_video, heatmap_video, masked_heatmap_video]
    elif args.output_mode=='infer':
        video_list = [nmfc_video, eye_gaze_video, fake_video]
    else:
        if eye_gaze_video is not None:
            video_list = [rgb_video, nmfc_video, eye_gaze_video, fake_video, heatmap_video, masked_heatmap_video]
        else:
            video_list = [rgb_video, nmfc_video, fake_video, heatmap_video, masked_heatmap_video]
    # write
    if args.output_mode == 'source_target_separate':
        write_video_to_file(save_path[:-4] + '_source' + '.mp4', video_list[0])
        write_video_to_file(save_path[:-4] + '_target' + '.mp4', video_list[1])
    elif args.output_mode == 'source_nmfc_target_seperate':
        source_video = video_list[0]
        source_video_save_path = os.path.join(path, args.save_video_name)
        write_video_to_file( source_video_save_path + '_source' + '.mp4', source_video)

        input_video = np.concatenate(video_list[1:3], axis=2)
        input_video_save_path = os.path.join(path, args.save_video_name)
        write_video_to_file( input_video_save_path + '_input' + '.mp4', input_video)

        target_video = video_list[-1]
        target_video_save_path = os.path.join(path, args.save_video_name)
        write_video_to_file( target_video_save_path + '_target' + '.mp4', target_video)

        final_video = np.concatenate(video_list, axis=2)
        write_video_to_file(save_path, final_video)
    else:
        final_video = np.concatenate(video_list, axis=2)
        write_video_to_file(save_path, final_video)

if __name__=='__main__':
    main()
