from .utils import Atom, Residue, ActiveSite
import numpy as np

def get_backbone_coords(residue):
    """just a getter for a tuple of backbone coordinates"""
    return [atom.coords for atom in residue.atoms[:3]]

def euclidean_dist(r1, r2):
    """calculates the euclidean distance between two residues"""
    # there's three xyzs in each coord tuple, corresponding to the n,
    # ca, c backbone atoms. We calculate the distance between n1 and
    # n2, ca1 and ca2, c1 and c2; and sum those distances.
    coords1 = get_backbone_coords(r1)
    coords2 = get_backbone_coords(r2)
    s = 0
    for i in range(3):
     s = s + (((coords1[i][0] - coords2[i][0])**2 +
        (coords1[i][1] - coords2[i][1])**2 + 
        (coords1[i][2] - coords2[i][2])**2)**(0.5))
    return s

def compute_similarity(site_a, site_b):
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    # I want to find the minimum distance between each
    # residue in each active site and any residue in the other active
    # site, and take the sum of all those distances. This does a good
    # job of fully capturing the distances between sites in a
    # symmetric way. Because sites with more residues will sum more
    # distances, I then normalize by the number of distances I summed.
    # This metric is pretty nice for capturing a wide variety of
    # situations: if the active sites are very far apart (non-
    # overlapping), this is just the distance between every residue
    # and the one residue closest the other site; if they're close,
    # then this records well how close each residue is to its
    # overlapping partner.
    similarities = [] # we're keeping a running sum.
    # first, looping over site_a:
    for residue_a in site_a.residues:
        # find the closest residue in b
        closest = (float('Inf'), None)
        for residue_b in site_b.residues:
            thisDist = euclidean_dist(residue_a, residue_b)
            if thisDist < closest[0]:
                closest = (thisDist, residue_b)
        # then just add it up and move along.
        similarities.append(closest[0])
    # likewise, looping over site_b:
    for residue_b in site_b.residues:
        # find the closest residue in a
        closest = (float('Inf'), None)
        for residue_a in site_a.residues:
            thisDist = euclidean_dist(residue_a, residue_b)
            if thisDist < closest[0]:
                closest = (thisDist, residue_a)
        # then just add it up and move along.
        similarities.append(closest[0])
    # finally, norm by the number of residues we're summing over.
    return sum(similarities)/len(similarities)

def calc_distance_matrix(active_sites):
    """
    returns a dict of distances given a list of active sites.
    """
    distD = {} # keyed by (site_a, site_b) tuple; value a dist
    for i in range(len(active_sites)):
        for j in range(i, len(active_sites)):
            site_a = active_sites[i]
            site_b = active_sites[j]
            distD[frozenset([site_a, site_b])] = compute_similarity(site_a, site_b)
    return distD

def closestMedioidI(active_site, medioids, distD):
    """
    returns the index of the closest medioid in medioids to active_site

    input: active_site, an ActiveSite instance
           medioids, a list of ActiveSite instances
           distD, a dictionary of distances
    output: the index of the ActiveSite closest to active_site in medioids
    """
    closest = (float('Inf'), None)
    for i, medioid in enumerate(medioids):
        thisDist = distD[frozenset([active_site, medioid])]
        if thisDist < closest[0]:
            closest = (thisDist, i)
    return closest[1]

def totalCost(medioids, assignments):
    """
    gets the cost of assigning a list of lists of ActiveSites(assignments)
    to a corresponding list of medioids.
    """
    runningSum = 0
    for medioid, assignment in zip(medioids, assignments):
        runningSum += sum([distD[frozenset([site, medioid])] for site in assignment])
    return runningSum

def getAssignments(medioids, active_sites, distD):
    """
    assigns active sites to medioids

    input: active_sites, a list of all ActiveSites to assign
           medioids, a subset of that list to assign to
    output: a list of sets of active_sites assigned to each
            corresponding(by index) 
    """
    assignments = [set() for medioid in medioids]
    for site in active_sites:
        assignments[closestMedioidI(site, medioids, distD)].add(site)
    return assignments

MEANS = 3
def cluster_by_partitioning(active_sites):
    """
    Cluster a given set of ActiveSite instances using PAM

    In PAM (partitioning around medioids), random cluster medioids are
    chosen from the points to be clustered; all points are assigned to
    the nearest medioid; then, iteratively, swaps of all non-medioid
    points with each mediod are probed to find lower-cost clusterings
    until none can be found. Empirically, this greedy method has been
    shown to probe a broader search space of clusterings than K-means
    and also doesn't require recalculation of distances.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # calculate entire distance matrix once
    sys.stderr.write("calculating distances...\n")
    distD = calc_distance_matrix(active_sites)
    # pick initial medioids
    medioids = np.random.choice(active_sites, MEANS, replace=False)
    sys.stderr.write("initial cost %s." % \
        totalCost(medioids, getAssignments(medioids, active_sites, distD)))
    # loop, finding new medioids and reassigning
    loopNum = 0
    changes = float("inf")
    while changes > 0:
        # in each iteration, loop through the medioids. Stop when we
        # make no changes.
        changes = 0
        loopNum += 1
        sys.stderr.write("loop %s... " % loopNum)
        for i, medioid in enumerate(medioids):
            # For each medioid, try swapping with every non-medioid site
            # and see if the cost decreases. We keep track of the best
            # swap and if it's better than the total cost, we swap.
            # first, we have to cost out the current assignment.
            # it might seem like we shouldn't do this in the loop, but
            # remember that we're modifying the list of medioids as we
            # iterate.
            assignments = getAssignments(medioids, active_sites, distD)
            bestSwap = (totalCost(medioids, assignments), medioids)
            # loop through non-medioid sites
            for test_site in active_sites:
                if test_site not in medioids:
                    # create the new list of medioids and corresponding
                    # potentially-different cluster assignments
                    test_medioids = medioids.copy()
                    test_medioids[i] = test_site
                    test_assignments = getAssignments(test_medioids, active_sites, distD)
                    # then cost it
                    test_cost = totalCost(test_medioids, test_assignments)
                    # and see if it's better.
                    if test_cost < bestSwap[0]:
                        bestSwap = (test_cost, test_medioids)
            # all right, we've found the best swap. If we improved, we
            # keep it and increment the number of changes we've made
            # this round. Useful to note here that, while we're
            # modifying the list we're iterating over, it's by swapping
            # the element under the iterator, and never modifying the
            # list past the iterator, so this is safe.
            if not np.array_equal(medioids, bestSwap[1]):
                changes += 1
                medioids = bestSwap[1]
        # done iterating through every medioid this round...
        sys.stderr.write("%s swaps made. current cost %s.\n" % (changes, bestSwap[0]))
    # done with our while loop. We went through a round of trying swaps
    # and didn't make any changes, so we're done
    sys.stderr.write("done.\n")
    return getAssignments(medioids, active_sites, distD)

linkageFunction = max
STOP_K = 3
def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #

    Input: a list of ActiveSite instances
    Output: a single clustering (a slice through the hierarchical clustering) at
    width STOP_K.
    """
    # Fill in your code here!
    # get distances
    sys.stderr.write("calculating distances...\n")
    distD = calc_distance_matrix(active_sites)
    # I want to weed out the self-distances so I can take a nice min.
    for site in active_sites: del distD[frozenset([site])]
    # setD gives the cluster memebership for each site.
    clusters = set(active_sites)
    clusterD = {}
    # initialize clusters. They're named according to their seed
    # site. This is super convenient because our site distance dict
    # suddenly just became our cluster distance dict!
    for site in active_sites:
        clusterD[site] = [site]
    # we're going to iterate until we have as many clusters as STOP_K.
    iterNum = 0
    while len(clusters) > STOP_K:
        iterNum += 1
        sys.stderr.write("step %s...\n" % iterNum)
        # find closest pair of clusters; they'll form the new cluster. It
        # will have the same name as cluster_a.
        cluster_a, cluster_b = min(distD, key=distD.get)
        print (cluster_a, cluster_b)
        clusters.remove(cluster_a)
        clusters.remove(cluster_b)
        distD.pop(frozenset([cluster_a, cluster_b]))
        # reassign cluster membership for all their elements
        clusterD[cluster_a] = clusterD[cluster_a] + clusterD[cluster_b]
        del clusterD[cluster_b]
        # replace the distances to the two old clusters with the dist to
        # the new one
        for otherCluster in clusters:
            dist_a = distD.pop(frozenset([cluster_a, otherCluster]))
            dist_b = distD.pop(frozenset([cluster_b, otherCluster]))
            dist_new = linkageFunction(dist_a, dist_b)
            distD[frozenset([cluster_a, otherCluster])] = dist_new
        clusters.add(cluster_a)
    return clusterD
