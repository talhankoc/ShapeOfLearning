import imageio

path = 'Homology/Data/CIFAR-10-AML/'
folder_prefix = 'model-'
epochs = 59
images1 = []
images2 = []
for epoch in range(1,epochs+1):
	folder_path = path + folder_prefix + str(epoch) +'/'
	img1name = folder_path + 'birth_death.png'
	img2name = folder_path + 'lifetime.png'
	images1.append(imageio.imread(img1name))
	images2.append(imageio.imread(img2name))
imageio.mimsave(path + 'Graphs/birthDeath.gif', images1, duration=float(1/3))
imageio.mimsave(path + 'Graphs/lifetime.gif', images2, duration=float(1/3))

