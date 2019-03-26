

def run(loadPaths, savePath):
	import imageio
	import os
	images = []
	print('GIF Make: loading images...')
	for path in loadPaths:
		images.append(imageio.imread(path))

	print('GIF Make: saving gif files...')	
	imageio.mimsave(savePath, images, duration=float(1/3))

