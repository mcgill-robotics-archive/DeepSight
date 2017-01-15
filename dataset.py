class dataset(object):

	def __init__(self, arg):
		super(dataset, self).__init__()
		self.num_imgs = 0
		self.image_dims = (0,0,0)
		self.batch_size = 0
		self.curr_batch = 0
		
	def open_folder(folder):
		return folder

	def get_all_images():
		return all_images

	def set_batch_size(size):
		if self.curr_batch != 0:
			raise RuntimeError("Cannot change the batch size while iterating. Run reset_batch() then try to change the batch size!")
		else:
			self.batch_size = size

	def get_next_batch():
		return next_batch

	def reset_batch():
		self.curr_batch = 0