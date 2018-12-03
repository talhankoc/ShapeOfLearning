import imageio

path = '/Users/tkoc/Code/ShapeOfLearning/Homology/FloydVRInputLayer/'
gif_path = path + 'GIFs/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'Fashion2_'
layer_sizes = [8,16,24,32,40,48]
epochs = 50

for layer_size in layer_sizes:
	images1 = []
	images2 = []
	for epoch in range(1, epochs+1):
		folder_path = path + folder_prefix + str(epoch) + '_' + str(layer_size) + '/'
		img1name = folder_path + 'birth_death.png'
		img2name = folder_path + 'lifetime.png'
		images1.append(imageio.imread(img1name))
		images2.append(imageio.imread(img2name))
	print('Saving gifs for layer', layer_size, '...')
	imageio.mimsave(gif_path + 'birthDeath_layerSize' + str(layer_size) + '.gif', images1, duration=float(1/3))
	imageio.mimsave(gif_path + 'lifetime_layerSize' + str(layer_size) + '.gif', images2, duration=float(1/3))
