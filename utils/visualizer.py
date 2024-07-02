### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import time
from . import util
from . import html_
# import scipy.misc
from PIL import Image
import tensorflow as tf


import torch
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
# tensorboard --logdir_spec=h1:"log3/runs01/",h2:"log4/runs01/" --port=6006 --bind_all
class Visualizer():
    # lxp
    # 增加验证集loss可视化
    # def __init__(self, opt):
    def __init__(self, opt, is_val = False):
        self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        self.tf = tf

        # lxp
        # self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        if is_val:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'val_logs')
        else:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')


        if opt.isTrain:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)
            physical_devices = tf.config.experimental.list_physical_devices('CPU')
            self.writer = tf.summary.create_file_writer(self.log_dir)# tf.summary.FileWriter(self.log_dir),old version
            self.writer_loss_D = tf.summary.create_file_writer(os.path.join(self.log_dir,'loss_D'))
            self.writer_loss_G = tf.summary.create_file_writer(os.path.join(self.log_dir,'loss_G'))

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        if opt.isTrain:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                # scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Image.fromarray((image_numpy)).save(s,format="jpeg")
                image_tensor = torch.tensor(image_numpy).unsqueeze(0)
                image_name = label

                # Create an Image object
                # img_sum = self.tf.summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                with self.writer.as_default():
                    self.tf.summary.image(image_name,image_tensor,step,max_outputs=3)
                    self.writer.flush()

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html_.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 6:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):

        # tf.summary.scalar(tag, value, step=step)
        # self.writer.flush()
        
        for tag, value in errors.items():
            if tag == 'G_total_loss' :
                with self.writer_loss_G.as_default():
                    self.tf.summary.scalar(tag, value,step=step)
                    self.writer_loss_G.flush()
            elif tag == 'D_total_loss':
                with self.writer_loss_D.as_default():
                    self.tf.summary.scalar(tag, value,step=step)
                    self.writer_loss_D.flush()
            else:
                with self.writer.as_default():
                    self.tf.summary.scalar(tag, value,step=step)
                    self.writer.flush()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) \n' % (epoch, i, t)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.7f \n' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path, webpage=None, iter = 0):
        dirname = os.path.basename(os.path.dirname(image_path[0]))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = os.path.basename(image_path[iter])
        name = os.path.splitext(name)[0]

        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            util.mkdir(os.path.join(image_dir, label))
            image_name = '%s.%s' % (name, 'png')
            save_path = os.path.join(image_dir, label, image_name)
            print("saving_images:", save_path)
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)
    def save_images_(self, image_dir, visuals, image_path, webpage=None):
        dirname = os.path.basename(os.path.dirname(image_path))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = os.path.basename(image_path)
        name = os.path.splitext(name)[0]


        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            util.mkdir(os.path.join(image_dir, label))
            image_name = '%s.%s' % (name, 'png')
            save_path = os.path.join(image_dir, label, image_name)
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)

    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
