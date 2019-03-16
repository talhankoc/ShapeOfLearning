
def makeGif(folder_root_path, folder_data_path_prefix, epoch_range):
	import imageio
	import os
	images1 = []
	images2 = []
	print('GIF Make: loading images...')
	for epoch in epoch_range:
		curr_path = folder_root_path + folder_data_path_prefix + str(epoch) +'/'
		img1name = curr_path + 'birth_death.png'
		img2name = curr_path + 'lifetime.png'
		images1.append(imageio.imread(img1name))
		images2.append(imageio.imread(img2name))
	try:
		os.makedirs(savePath + symbName)
	except:
		print('Graph folder already exists')
	print('GIF Make: saving gif files...')	
	imageio.mimsave(folder_root_path + 'Graphs/birthDeath.gif', images1, duration=float(1/3))
	imageio.mimsave(folder_root_path + 'Graphs/lifetime.gif', images2, duration=float(1/3))

path = 'Homology/Data/CIFAR-10-Variation2/'
folder_prefix = 'model-'
epochs = range(101)

makeGif(path, folder_prefix, epochs)