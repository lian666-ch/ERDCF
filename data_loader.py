import os
import numpy as np
from PIL import Image
import math

def read_file_list(filelist):

	pfile = open(filelist)
	filenames = pfile.readlines()
	pfile.close()
	filenames = [f.strip() for f in filenames]

	return filenames

def split_pair_names(filenames, base_dir):

	filenames = [c.split(' ') for c in filenames]
	filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

	return filenames


class DataParser():

	def __init__(self, batch_size_train):
		self.batch_size_train = batch_size_train
		self.test_file = os.path.join('./dataset/', 'test.txt')
		self.test_data_dir = './dataset/DeepCrack/'
		self.testing_pairs = read_file_list(self.test_file)
		self.test_samples = split_pair_names(self.testing_pairs, self.test_data_dir)
		self.test_n_samples = len(self.testing_pairs)
		self.test_ids = np.arange(self.test_n_samples)
		self.test_steps = math.ceil(len(self.test_ids)/batch_size_train)

		self.train_file = os.path.join('./dataset/', 'train.txt')
		self.train_data_dir = './dataset/DeepCrack/'
		self.training_pairs = read_file_list(self.train_file)
		self.samples = split_pair_names(self.training_pairs, self.train_data_dir)
		self.n_samples = len(self.training_pairs)
		self.train_ids = np.arange(self.n_samples)
		self.train_steps = math.ceil(len(self.train_ids)/batch_size_train)


	def get_batch(self, batch):
		images = []
		edgemaps = []
		threshold =0
		for idx, b in enumerate(batch):
			im = Image.open(self.samples[b][0])
			em = Image.open(self.samples[b][1])

			new_img = np.array(im, dtype=np.float32)
			new_em = np.array(em.convert('L'), dtype=np.float32)

			bin_em = new_em
			bin_em /= 255.0
			bin_em[bin_em <= threshold] = 0
			bin_em[bin_em > threshold] = 1

			# Some edge maps have 3 channels some dont
			bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
			# To fit [batch_size, H, W, 1] output of the network
			bin_em = np.expand_dims(bin_em, 2)

			images.append(new_img)
			edgemaps.append(bin_em)

		images   = np.asarray(images)
		edgemaps = np.asarray(edgemaps)

		return images, edgemaps


