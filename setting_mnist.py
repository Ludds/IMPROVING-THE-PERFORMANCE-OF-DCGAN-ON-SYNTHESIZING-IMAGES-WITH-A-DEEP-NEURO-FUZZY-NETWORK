class setting():
	# Params connected to input
	height_data=28
	width_data=28
	n_class=2
	depth=1
	latent=100
	# Hyper params
	gen_lr=0.0001	# 0.0001
	disc_lr=0.0001	# 0.001
	batch_size=16	# 16
	epsilon=1e-6	# 1e-6
	num_epoch=10000	# 10000
	beta = 0.5		# 0.5
	dropout = 0.75	# 0.75