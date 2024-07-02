from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import random

class InferenceDataset(Dataset):
	# pose_img
	# referece_img
	# gt_img
	def __init__(self, pose_root, gt_root,ld_root,opts,index_range,target_transform=None, source_transform=None,pose_transform=None,random = True):
		self.pose_paths = np.array(sorted(data_utils.make_dataset(pose_root)))
		self.target_paths = np.array(sorted(data_utils.make_dataset(gt_root)))
		if ld_root!= None:
			self.ld_paths = np.array(sorted(data_utils.make_dataset(ld_root)))
		index_range=np.array(index_range)
		# index_range = index_range*50
		self.opts = opts
		hardcase = self.opts.hardcase
		index_range[1] = len(self.target_paths)-1 if index_range[1]>=len(self.target_paths) else index_range[1]
  # self.ref_paths = sorted(data_utils.make_dataset(gt_root))
		if hardcase == True:
			hardcase = np.load("/home/zt/lxp/hyperstyle/pretrained_models/yuchen_quanshen.npy")
			# print(hardcase)
			total = []
			for hd in hardcase:
				if hd[0]>index_range[1] or hd[1]>index_range[1] or hd[1] < index_range[0] or hd[0] < index_range[0]:
					continue			
				for i in range(hd[0],hd[1]+1):
					total.append(i)
			print(len(total))
			self.target_paths = self.target_paths[total]
			self.pose_paths = self.pose_paths[total]
			# self.ld_paths = self.ld_paths[total]
			# ref_paths = ref_paths[total]
		else:
			self.pose_paths = self.pose_paths[index_range[0]:index_range[1]+1]
			self.target_paths = self.target_paths[index_range[0]:index_range[1]+1]
			# self.ld_paths = self.ld_paths[index_range[0]:index_range[1]]
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.pose_transform = pose_transform
		if random:
			self.random_img= random.randint(0,len(self.target_paths))
		else:
			self.random_img= 0
	def set_random_img(self):
		self.random_img= random.randint(0,len(self.target_paths))
  
	def __len__(self):
		return len(self.target_paths)
	# use the source image of eyery 200 imgs
	def __getitem__(self, index):
		pose_paths = self.pose_paths[index]
		target_paths = self.target_paths[index]
		ref_path = self.target_paths[self.random_img]
  
		pose_im = Image.open(pose_paths).convert('RGB')
		gt_im = Image.open(target_paths).convert('RGB')
		ref_im = Image.open(ref_path).convert('RGB')
  
		if self.target_transform:
			gt_im = self.target_transform(gt_im)
			org_pose = pose_im.copy()
			org_pose = self.target_transform(org_pose)
		if self.pose_transform:
			pose_im = self.pose_transform(pose_im)
		if self.source_transform:
			ref_im = self.source_transform(ref_im)
		# else:
		# 	pose_im = gt_im
		 #ref #target #pose
		return ref_im,gt_im,pose_im,org_pose
