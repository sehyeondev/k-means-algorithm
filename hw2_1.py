import sys
import math
from pyspark import SparkConf, SparkContext

# get distance of two points
def get_distance(l1, l2):
    sumsq = 0
    for i in range(len(l1)):
        sumsq += (float(l1[i]) - float(l2[i])) **2
    return math.sqrt(sumsq)

# initialize k points
def initialize_k_points(input_file, k_value):
    init_points = [0] # pick first point as first point in the dataset
    # file = open(input_file, "r")
    with open(input_file) as file:
        lines = file.read().splitlines()
        while len(init_points) < k_value:
            max_dis = [10**9, 0]
            for i in range(len(lines)):
            # i = 0
            # for line in file.readlines():
                cand_point = lines[i].split()
                min_dis = [10**9, 10**9]
                for j in init_points:
                    if i in init_points:
                        continue
                    dis = get_distance(cand_point, lines[j].split())
                    if dis < min_dis[1]:
                        min_dis[0] = i
                        min_dis[1] = dis
                    if min_dis[1] > max_dis[1]:
                        max_dis[0] = min_dis[0]
                        max_dis[1] = min_dis[1]
                # i += 1
        init_points.append(max_dis[0])

    return init_points
    
  
    return init_points

# get distances from one point to k clusters
def get_distances(line, k_init_features):
    point_features = line.split()
    distances = []
    for i in range(k_init_features):
        distances.append((i, get_distance(k_init_features[i], point_features)))
    return distances

# get closest cluster of each point
def get_closest_cluster(k_dis):
    select = 0
    for i in range(1, len(k_dis)):
        if k_dis[select][1] > k_dis[i][1]:
            select = i
    return (k_dis[i])

# k-means algorithm
def k_means(input_file, k_init_points):
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(input_file)
    k_init_features = [lines[c_num].split for c_num in k_init_points]
    all_distances = lines.map(lambda line: get_distances(line, k_init_features)) # (c_num, dis)
    final_clusters = all_distances.flatMap(lambda k_dis: get_closest_cluster(k_dis))
    return final_clusters

# main function
# [1] get input_file and k-value from command line argument
# [2] initialize k clusters
# [3] cluster remaining points
# [4] get diameter of k clusters
# [5] print average diameter


def main():
    input_file = sys.argv[1] # get input file
    # print (type(sys.argv[2]))
    k_value = int(sys.argv[2]) # get k-value
    k_init_points = initialize_k_points(input_file, k_value) # initialize k clusters
    print (k_init_points)
    clusters = k_means(input_file, k_init_points) # cluster remaining points <- k-means algorithm
    print (clusters.collect())
    diameters = clusters.reduceBykey(lambda d_x, d_y: max(d_x, d_y)) # get diameter of all clusters
    print (diameters.collect())
    avg_diameter = float(sum(diameters))/float(len(diameters)) # get average diameter
    print (avg_diameter)

if __name__ == '__main__':
    main()