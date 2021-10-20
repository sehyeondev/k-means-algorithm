import sys
from pyspark import SparkConf, SparkContext

## ======================================
## basic functions for clustering
## get_distance(point1, point2)
## get_closest cluster(point_feature, k_init_features)
## ======================================
# get distance between two points
def get_distance(point1, point2):
    sumsq = 0
    for i in range(len(point1)):
        sumsq += (float(point1[i]) - float(point2[i])) **2
    return sumsq**0.5

# get closest cluster for one point
def get_closest_cluster(k_init_features, point):
    point_feature = point.split()
    distances = [get_distance(point_feature, init_f) for init_f in k_init_features]
    result = [distances.index(min(distances)), point_feature]
    return result

## ============================================
## initialize k points -> center of k clusters
## cluster remaining points
## ============================================
def initialize_k_points(input_file, k_value):
    init_points = [0] # pick first point as first point in the dataset
    file = open(input_file, "r")
    lines = file.readlines()
    file.close()
    
    while len(init_points) < k_value:
        find_max = []
        for i in range(len(lines)):
            cand_point = lines[i].split()
            find_min = []
            for j in init_points:
                find_min.append(get_distance(cand_point, lines[j].split()))
            min_dis = min(find_min)
            find_max.append(min_dis)
        max_dis = max(find_max)
        init_points.append(find_max.index(max_dis))
    
    return init_points
# cluster remaining points
def k_means(input_file, k_init_points):
    # make lines as list
    file = open(input_file, "r")
    lines_list = file.readlines()
    file.close()
    # make lines for map-reduce
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(input_file)
    # cluster remaining points
    k_init_features = [lines_list[c_num].split() for c_num in k_init_points]
    cluster_pairs = lines.map(lambda point: get_closest_cluster(k_init_features, point))
    clusters = cluster_pairs.groupByKey().mapValues(list)
    # .collect()
    return clusters

## ==================================================
## after clustering all points by k-means algorithm
## find diameter of each cluster and average diameter
## diameter: largest distance of any two points
## ==================================================
# get diameter from points in a cluster
def get_diameter(points):
    diameter = 0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dis = get_distance(points[i], points[j])
            if dis > diameter:
                diameter = dis
    print (diameter)
    return diameter

# get average diameter of all clusters
def get_avg_diameter(diameter_pairs):
    sum_diameters = 0
    for pair in diameter_pairs:
        sum_diameters += pair[1]
    avg_diameter = sum_diameters/len(diameter_pairs) # get average diameter
    return avg_diameter

## ==========================================================
## main function
## [1] get input_file and k-value from command line argument
## [2] initialize k clusters
## [3] cluster remaining points
## [4] get diameter of k clusters
## [5] print average diameter
## ==========================================================
def main():
    input_file = sys.argv[1] # get input file
    k_value = int(sys.argv[2]) # get k-value
    k_init_points = initialize_k_points(input_file, k_value) # initialize k clusters
    print("k_init_points")
    print (k_init_points)
    clusters = k_means(input_file, k_init_points) # cluster remaining points <- k-means algorithm
    # print ("clusters")
    # print (clusters.collect())
    diameter_pairs = clusters.mapValues(lambda points: get_diameter(points)).collect()
    print("diameter_pairs")
    print (diameter_pairs)
    avg_diameter = get_avg_diameter(diameter_pairs)
    print ("avg_diameter")
    print (avg_diameter)

if __name__ == '__main__':
    main()