import pickle
from numpy import inf
from BettiPlotter import plot_dgms

#-----------------------------------------#
path = 'Homology/Data/CIFAR-10-Variation2/'
file_name = 'VRFiltration_BettiData.txt'
folder_prefix = 'model-'
epochs = range(101)

def analysis(data):
    lifetime = np.array([ [x,y-x] for x,y in data if y != inf ])
    mean, std = np.mean(lifetime[:,1]), np.std(lifetime[:,1])
    above_standard_deviation_points = np.array([point for point in lifetime if point[1] > mean + std ])
    return lifetime, mean, std, above_standard_deviation_points

for epoch in epochs:
    print('Epoch', epoch)
    folder_path = path + folder_prefix + str(epoch) +'/'
    fn = folder_path + file_name
    x_range = 25.0
    with open(fn, 'rb') as f:
        diagrams = pickle.load(f)
        data_betti1 = diagrams[1]

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data_betti1)

        h1_lifetime = [ [y-x, x] for x,y in diagrams[0] if y != inf ]


        plot_dgms(diagrams, 
            size=12,
            title='CIFAR-10-3_Layer | CNN |'+' Epoch='+str(epoch),
            save_path=folder_path+'birth_death.png',
            xy_range=[-1,x_range,-1,x_range]
            )
        plot_dgms(diagrams, 
            size=12, 
            title='CIFAR-10-3_Layer | CNN |'+' Epoch='+str(epoch),
            save_path=folder_path+'lifetime.png', 
            xy_range=[-1,x_range,-1,x_range], 
            lifetime=True
            )
        with open(folder_path + 'analysis.txt', 'wb') as f2:
            # TODO incorporate analysis into this

            h0_total = len(diagrams[0])
            h1_total = len(diagrams[1])
            avg_h0_life = sum([ y-x for x,y in diagrams[0] if y != inf ])/h0_total
            avg_h1_life = sum([ y-x for x,y in diagrams[1] if y != inf ])/h1_total
            pickle.dump([h0_total,h1_total, avg_h0_life,avg_h1_life], f2)

    
