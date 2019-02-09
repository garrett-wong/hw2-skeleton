import sys
from .io import read_active_sites, write_clustering, write_mult_clusterings
from .cluster import cluster_by_partitioning, cluster_hierarchically, calc_distance_matrix, labelClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

active_sites = read_active_sites(sys.argv[2])

# calculate entire distance matrix once
print("calculating distances...")
distD = calc_distance_matrix(active_sites)

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    clustering = cluster_by_partitioning(active_sites, distD, 3)
    write_clustering(sys.argv[3], clustering)

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    clusterings = cluster_hierarchically(active_sites, distD)
    write_mult_clusterings(sys.argv[3], clusterings)

if sys.argv[1][0:2] == "-E":
	print("Evaluating both clustering methods")
	# We need our distances as an array instead of a dict
	distArray = []
	for site_a in active_sites:
		thisRow = []
		for site_b in active_sites:
			thisRow.append(distD[frozenset([site_a, site_b])])
		distArray.append(thisRow)
	# First, get the silhouette for PAM with k from 2 to 20
	print("Trying PAM with k from 2..20")
	partitioning_silhouettes = []
	for k in range(2, 21):
		partitioning = cluster_by_partitioning(active_sites, distD, k)
		# and our partitioning as a set of labels instead of a list of
		# clusters.
		labels = labelClustering(partitioning, active_sites)
		partitioning_silhouettes.append(silhouette_score(distArray, labels))
	# Now do the same for hierarchical clustering...
	print("Evaluating hierarchical clustering with 2..20 clusters...")
	hierarchical_silhouettes = []
	hierarchy = cluster_hierarchically(active_sites, distD)
	for hClustering in hierarchy[-2:-21:-1]:
		labels = labelClustering(hClustering, active_sites)
		hierarchical_silhouettes.append(silhouette_score(distArray, labels))
	# and finally, for randomly assigned labels.
	print("Generating random labels...")
	random_silhouettes = []
	for i in range(2, 21):
		labels = np.random.choice(range(i), len(active_sites), replace=True)
		random_silhouettes.append(silhouette_score(distArray, labels))
	# Then plot.
	print("plotting...")
	silPlot = plt.figure()
	plt.plot(range(2, 21), partitioning_silhouettes, label = "PAM (partitioning)")
	plt.plot(range(2, 21), hierarchical_silhouettes, label = "hierarchical")
	plt.plot(range(2, 21), random_silhouettes, label="random labelling")
	plt.xlabel("number of clusters")
	plt.ylabel("silhouette (higher is better)")
	plt.xticks(range(2, 21))
	plt.title("choosing K")
	plt.legend()
	#plt.show()
	silPlot.savefig('silhouette.png')
	print("done.")

if sys.argv[1][0:2] == "-C":
	print("comparing both clustering methods")
	hierarchy = cluster_hierarchically(active_sites, distD)
	overlaps_counts = []
	rp_overlap = []
	rh_overlap = []
	for k in range(2, 21):
		print(k)
		partitioning = cluster_by_partitioning(active_sites, distD, k)
		hClustering = hierarchy[-k]
		# we also make a random clustering. This is kind of kludgey
		# because, randomly adding sites to clusters, we have to make
		# sure we don't end up with a cluster with zero sites.
		randomClust = [[] for x in range(k)]
		while min([len(x) for x in randomClust]) == 0:
			randomClust = [[] for x in range(k)]
			for site in active_sites:
				randomClust[np.random.choice(range(k))].append(site)
		print(len(partitioning), len(hClustering), len(randomClust))
		overlaps_counts.append(compareClusterings(partitioning, hClustering))
		rp_overlap.append(compareClusterings(randomClust, partitioning))
		rh_overlap.append(compareClusterings(randomClust, hClustering))
		overlapPlot = plt.figure()
	plt.plot(range(2, 21), overlaps_counts, label="PAM and hierarchical clustering overlap")
	plt.plot(range(2, 21), rp_overlap, label="random and PAM clustering overlap")
	plt.plot(range(2, 21), rh_overlap, label="random and hierarchical clustering overlap")
	plt.xlabel("number of clusters")
	plt.ylabel("summed best overlaps")
	plt.title("evaluating overlap between PAM and hierarchical clustering")
	plt.xticks(range(2, 21))
	plt.legend()
	#plt.show()
	overlapPlot.savefig("overlap.png")
	print("done.")