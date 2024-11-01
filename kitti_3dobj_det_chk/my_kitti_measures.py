import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from   matplotlib.path import Path
from matplotlib import colors
import numpy as np
from PIL import Image
from math import sin, cos
import argparse
import open3d as o3d

#my imports
#for optics
import matplotlib.gridspec as gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
#for hdbcan
from sklearn import cluster
import time
#for pointcloud density
from sklearn.neighbors import NearestNeighbors
#for bounding box
import math
import itertools


dev_velodyne_path =  "C:/Users/Havasi/source/repos/resources/image_02_for_dev/velodyne/"
dev_velo_path = "C:/Users/Havasi/Desktop/raw/2011_09_26_drive_0104_sync/2011_09_26/2011_09_26_drive_0104_sync/velodyne_points/data/"

dev_tosave_path = "./GT/"

def pointsFromPcd(pcd):
  points = np.asarray(pcd.points)
  colors = pcd.colors
  return points, colors

def pointsFromVelo(velo):
  return np.asarray(velo[:,:3])

def pcdFromPoints(points, pcd_colors = None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  if(pcd_colors != None): 
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
  return pcd

def showPcd(pcds):
  """
  pcds: [pcd] or [pcds] 
  """
  o3d.visualization.draw_geometries(pcds)  

def getFrames():
  return os.listdir(dev_velo_path)


def loadVeloFiles (frame):
  # load velodyne file 
  fn = dev_velo_path + frame
  velo = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
  
  return velo 

def lookupFilesInDir(path):
  return os.listdir(path)

def pointcloud_density(points, _n_neighbors = 2, _algorithm='ball_tree'):
  nbrs = NearestNeighbors(n_neighbors=_n_neighbors, algorithm=_algorithm).fit(points)
  distances, indices = nbrs.kneighbors(points)
  rho = np.average(distances[:,1])
  return rho, distances, nbrs

def histogram(points, idx, show = False):
  _ = plt.hist(points[:,idx], bins='auto')
  if(show): plt.show()

def down_sample(points, _smapling_ratio = 0.05, _every_k_points = 10, _voxel_size = 0.1, show = False):
  """
  """
  pcd = pcdFromPoints(points)
  random_pcd = pcd.random_down_sample(sampling_ratio=_smapling_ratio)
  uniform_pcd = pcd.uniform_down_sample(every_k_points=_every_k_points)
  voxel_pcd = pcd.voxel_down_sample(voxel_size=_voxel_size)

  if(show):
    showPcd([pcd])
    showPcd([random_pcd])
    showPcd([uniform_pcd])
    showPcd([voxel_pcd])
  
  return random_pcd,uniform_pcd,voxel_pcd

def outliers(points):
  pcd = pcdFromPoints(points)
  #voxel_pcd = pcd.voxel_down_sample(voxel_size=0.1)
  voxel_pcd = pcd

  #radius outlier removal
  pcd_rad, ind_rad = voxel_pcd.remove_radius_outlier(nb_points=16, radius=0.5)
  #display_inlier_outlier(voxel_pcd,ind_rad)

  #statistical outlier removal
  pcd_stat, ind_stat = voxel_pcd.remove_statistical_outlier(nb_neighbors=16,std_ratio=1.5)
  #display_inlier_outlier(voxel_pcd,ind_stat) 

  #return value shaping
  inlier_cloud_rad = voxel_pcd.select_by_index(ind_rad)
  inlier_points_rad = np.asarray(inlier_cloud_rad.points)
  inlier_cloud_stat = voxel_pcd.select_by_index(ind_stat)
  inlier_points_stat = np.asarray(inlier_cloud_stat.points)
  return inlier_points_rad, inlier_points_stat


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0., 0.8, 1.])
    showPcd([inlier_cloud, outlier_cloud])

def filter_pipeline(points):
  pcd = pcdFromPoints(points)
  
  #rho 
  rho, _, _ = pointcloud_density(points)

  #voxel down sample
  voxel_pcd = pcd.voxel_down_sample(voxel_size=rho*2)
  showPcd([voxel_pcd])

  #statistical outlier removal
  pcd_stat, ind_stat = voxel_pcd.remove_statistical_outlier(nb_neighbors=16,std_ratio=1.5)
  display_inlier_outlier(voxel_pcd,ind_stat) 

  #z axis histogram
  histogram(points, 2)
  #ground detection

  #crop bounding box 

def filter_pipeline_timings():
  filenames = lookupFilesInDir(dev_velo_path)

  hdbscan_time = np.array([])
  optics_time = np.array([])

  len_filedatas = np.array([])
  t_0 = time.time()

  hm = 5
  jl = 4
  step_eps = 0.3
  kl = 5
  step_sr = 0.1
    
  db_grid = np.zeros((hm,jl,kl)) 

  #for filename in filenames:
  for idx in range(hm):
    filename = filenames[idx+10]
    print(f'filename: {filename}')
    velo = loadVeloFiles(filename)
    points = np.asarray(velo[:,:3])
    len_filedatas = np.append(len_filedatas, len(points))
   
    for j in range(jl):
      print(f'j: {j}')
      for k in range(kl):
        print(k)
        eps = (j+1)*step_eps
        smpl_r = (k+1)*step_sr
        db_grid[idx][j][k] +=  dbscan(points, eps, smpl_r)

  r_grid = np.zeros((jl,kl))
  for i in range(kl):
    asd = np.zeros((hm,jl))
    asd = db_grid[:,:,i]
    r_grid[:,i] = np.sum(asd, axis = 0)
        
  print(f'db_grid: {db_grid}')

  r_mean_grid = np.zeros((jl,kl))
  for i in range(kl):
    asd = np.zeros((hm,jl))
    asd = db_grid[:,:,i]
    r_mean_grid[:,i] = np.mean(asd, axis = 0)

  print('r_mean_grid')
  print(r_mean_grid)

  r_std_grid = np.zeros((jl,kl))
  for i in range(kl):
    asd = np.zeros((hm,jl))
    asd = db_grid[:,:,i]
    r_std_grid[:,i] = np.std(asd, axis = 0)

  print('r_std_grid')
  print(r_std_grid)

  grid_titles = ['Szórás', 'Átlagos futási idő']
  grids = [r_std_grid, r_mean_grid]
  for w in range(2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ['r', 'g', 'b', 'y', 'c']
    yticks = range(kl)
    for c, k in zip(colors, yticks):
        #print(f'c: {c}, k: {k}, grid at[w][:,k]: {grids[w][:,k]}')
        xs = np.arange(jl)
        ys = grids[w][:,k]

        # You can provide either a single color or an array with the same length as
        # xs and ys. To demonstrate this, we color the first bar of each set cyan.
        cs = [c] * len(xs)
        #cs[0] = 'c'

        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel(f'eps, steps: {step_eps}')
    ax.set_ylabel(f'sampling_ratio, step: {step_sr}')
    ax.set_zlabel('times in s')
    ax.set_title(f'{grid_titles[w]}')

    # On the y-axis let's only label the discrete values that we have data for.
    ax.set_yticks(yticks)

    plt.show()

  avg_filedatas = np.mean(len_filedatas)
  print(avg_filedatas)
  std_filedatas = np.std(len_filedatas)
  print(std_filedatas)

  t_1 = time.time()
  print(f'all time: {(t_1 - t_0) / 60}')

    
#MyDBSCAN------------------------------------------------------------------
def mydbscan(points, eps = 0.5, _sampling_ratio = 0.5):
  pcd = pcdFromPoints(points)
  random_pcd = pcd.random_down_sample(sampling_ratio=_sampling_ratio)
  points = np.asarray(random_pcd.points)

  globl_cluster = np.zeros(len(points))

  #dbscan_thread()

'''
def dbscan_thread(globl_cluster, points, eps = 0.5, _sampling_ratio = 0.5):

  

  next_cluster_nr = 1
  clusters = np.zeros(len(points))

  for idx_x, vec_l in enumerate(points)-1:
    skip = False
    if(clusters[idx_x] == 0):
      clusters[idx_x] = next_cluster_nr
      next_cluster_nr += 1
      skip = True
    for idx_y, vec_r in enumerate(points)-1:
      if(clusters[idx_y+1] != 0):
        #
      else:
        #

        continue
      else:
        if(euclidean_dist(vec_l, vec_r) <= eps): clusters[idx_y+1] = clusters[idx_x]
'''


def euclidean_dist(v1,v2):
  return np.linalg.norm(v2-v1)


#DBSCAN---------------------------------------------------------------------
def dbscan(points, _eps = 0.5, _sampling_ratio = 0.5):
  print('dbscan')
  #rho, _, _= pointcloud_density(points)
  #print(rho)

  pcd = pcdFromPoints(points)
  random_pcd = pcd.random_down_sample(sampling_ratio=_sampling_ratio)
  points = np.asarray(random_pcd.points)

  t0 = time.time()
  with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=_eps, min_points=16, print_progress=True))
  t1 = time.time()
  t_delta = t1-t0

  '''
  max_label = labels.max()
  print(f"point cloud has {max_label +1 } clusters")
  colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
  colors[labels < 0 ] = 0
  pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
  showPcd([pcd])
  '''

  return t_delta

#OPTICS-----------------------------------------------------------------
def optics(points):
  print('optics')
  pcd = pcdFromPoints(points)
  print(len(points))

  #downsample
  #uniform_pcd = pcd.uniform_down_sample(every_k_points=10)
  random_pcd = pcd.random_down_sample(sampling_ratio=0.1)
  points = np.asarray(random_pcd.points)
  len_optics_points = len(points)

  #clust = OPTICS(min_samples=16, xi=0.05, min_cluster_size=0.05)
  clust = OPTICS(min_samples=16, xi=0.03)

  print("before fit")
  t0 = time.time()
  clust.fit(points)
  t1 = time.time()
  print("after fit")
  t_delta = t1-t0


  #cluster_optics_dbscan
  t2 = time.time()
  labels_050 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=0.5,
  )
  t3 = time.time()
  labels_100 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=1,
  )
  t4 = time.time()
  labels_200 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=2,
  )
  t5 = time.time()
  t_d_o_db_05 = t3 - t2
  t_d_o_db_1 = t4 - t3
  t_d_o_db_2 = t5 - t4

  '''
  space = np.arange(len(points[:,0]))
  reachability = clust.reachability_[clust.ordering_]
  print(reachability.shape)
  labels = clust.labels_[clust.ordering_]
  print(labels.shape)
  _ = plt.hist(labels, bins='auto')
  plt.show()

  plt.figure(figsize=(10, 7))
  G = gridspec.GridSpec(2, 4)
  ax1 = plt.subplot(G[0, :])
  ax2 = plt.subplot(G[1, 0])
  ax3 = plt.subplot(G[1, 1])
  ax4 = plt.subplot(G[1, 2])
  ax5 = plt.subplot(G[1, 3])

  # Reachability plot
  colors = ["g.", "r.", "b.", "y.", "c."]
  for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
  ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
  ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
  ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
  ax1.set_ylabel("Reachability (epsilon distance)")
  ax1.set_title("Reachability Plot")

  # OPTICS
  colors = ["g.", "r.", "b.", "y.", "c."]
  for klass, color in zip(range(0, 5), colors):
    Xk = points[clust.labels_ == klass]
    #print(Xk.shape)
    #print(Xk[:,:])
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
  ax2.plot(points[clust.labels_ == -1, 0], points[clust.labels_ == -1, 1], "k+", alpha=0.1)
  ax2.set_title("Automatic Clustering\nOPTICS")

  # DBSCAN at 0.5
  colors = ["g.", "r.", "b.", "c."]
  for klass, color in zip(range(0, 4), colors):
    Xk = points[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
  ax3.plot(points[labels_050 == -1, 0], points[labels_050 == -1, 1], "k+", alpha=0.1)
  ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

   # DBSCAN at 0.5
  colors = ["g.", "r.", "b.", "c."]
  for klass, color in zip(range(0, 4), colors):
    Xk = points[labels_100 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
  ax4.plot(points[labels_100 == -1, 0], points[labels_100 == -1, 1], "k+", alpha=0.1)
  ax4.set_title("Clustering at 1.0 epsilon cut\nDBSCAN")

  # DBSCAN at 2.
  colors = ["g.", "m.", "y.", "c."]
  for klass, color in zip(range(0, 4), colors):
    Xk = points[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
  ax5.plot(points[labels_200 == -1, 0], points[labels_200 == -1, 1], "k+", alpha=0.1)
  ax5.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")

  plt.tight_layout()
  plt.show()
  '''
  return t_delta, len_optics_points, t_d_o_db_05, t_d_o_db_1, t_d_o_db_2

#HDBSCAN----------------------------------------------------------------------
def hdbscan(points):
  print('hdbscan')
  pcd = pcdFromPoints(points)

  #downsample
  #uniform_pcd = pcd.uniform_down_sample(every_k_points=10)
  pcd = pcd.random_down_sample(sampling_ratio=0.1)
  points = np.asarray(pcd.points)
  len_hdbscan_points = len(points)


  '''min_samples: 
  The number of samples in a neighborhood for a point to be
  considered as a core point. This includes the point itself. 
  When None, defaults to min_cluster_size. '''
  '''min_cluster: 
  The minimum number of samples in a group for that group to be considered 
  a cluster; groupings smaller than this size will be left as noise.'''
  hdbscan = cluster.HDBSCAN(
    min_samples=10,
    min_cluster_size=40,
    allow_single_cluster=False,
  )
  print("begin fit")
  t0 = time.time()
  hdbscan.fit(points)
  t1 = time.time()
  t_delta = t1-t0
  print("end fit")
  print(f'fit time: {t_delta}')

  '''
  y_pred = hdbscan.labels_.astype(int)
  n_cluster = len(set(y_pred))
  max_label = hdbscan.labels_.max()
  print(f"point cloud has {max_label +1 } clusters")

  colors = plt.get_cmap("tab20")(y_pred / (n_cluster if n_cluster > 0 else 1))
  colors[y_pred < 0] = 0

  pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
  showPcd([pcd])
  '''

  return pcd, hdbscan, t_delta, len_hdbscan_points

def label_hdbscan(pcd, hdbscan):
  y_pred = hdbscan.labels_.astype(int)
  print('y_pred .shape', y_pred.shape)
  print('y_pred[:5]', y_pred[:5])
  n_cluster = len(set(y_pred))
  print('n_cluster', n_cluster)
  max_label = hdbscan.labels_.max()
  print(f"point cloud has {max_label +1 } clusters")
  cluster_element_numbers = np.zeros(n_cluster)
  print('len (y_pred)',len(y_pred))
  for i in range(len(y_pred)):
    cluster_element_numbers[y_pred[i]] += 1
  
  print('cluster_elemetns_numbers.shape', cluster_element_numbers.shape)
  print('c_e_n[:5]', cluster_element_numbers[:5])

  plt.bar(np.arange(len(cluster_element_numbers)), cluster_element_numbers, align='center')

  #plt.show()

  #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
  #az n legnagyobb szamossagu reszhalmaz idx-ei
  ne = 20
  n = ne if len(cluster_element_numbers) >= ne else len(cluster_element_numbers)-1
  print(f'n: {n}')
  print(type(n))
  rate = 1
  print(f'len(cluster_element_numbers): {len(cluster_element_numbers)}')
  idx_s = np.argpartition(cluster_element_numbers, int( rate * n))[int(-1 * rate * n):]
  print('idx_s:', idx_s)

  #top n sorted
  #idx_s = idx_s[np.argsort(cluster_element_numbers[idx_s])]
  #just index sort
  idx_s = np.sort(idx_s)
  print('idx_s_sort',idx_s)

  #top n elem szamossaga
  top_ten = cluster_element_numbers[idx_s]
  print('top10', top_ten)

  #voxel_pcd.select_by_index(ind_rad)
  pcds_list = []
  for i in range(len(idx_s)-1):
    acu = np.array([])
    search_value = idx_s[i]
    print('search_value',search_value)
    for j in range(len(y_pred)):
      if(y_pred[j] == search_value):
        acu = np.append(acu,j)
    #print('acu_type',type(acu))
    print('acu_len',len(acu))
    acu_list = acu.astype('i').tolist()
    #print('acu_list_type',type(acu_list))
    pcd_acu = pcd.select_by_index(acu_list)
    print('pcd_acu',pcd_acu)
    pcds_list = np.append(pcds_list,pcd_acu)
  
  for i in range(len(pcds_list)):
    aabb = pcds_list[i].get_axis_aligned_bounding_box()
    aabb.color = [1.,0., 0.]
    obb = pcds_list[i].get_oriented_bounding_box()
    obb.color = [1.,1.,0]
    #o3d.visualization.draw_geometries([pcds_list[i], aabb, obb])
  
  return pcds_list

def show_all(pcd, ground_pts, pcds_list):
  points = np.asarray(ground_pts[:,:3])
  ground_pcd = pcdFromPoints(points)

  # Number of points:
  n_points = points.shape[0]
  print('n_points', n_points)

  # Get the original points color to be updated:
  pcd_colors = np.zeros(shape=(n_points,3),dtype=float)
  print(pcd_colors.shape)

  for i in range(n_points):
    pcd_colors[i] = [0.,1.,0.]
  
  ground_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

  print('pcds_list_len',len(pcds_list))
  bb3d = []
  for i in range(len(pcds_list)):
    aabb = pcds_list[i].get_axis_aligned_bounding_box()
    aabb.color = [1.,0., 0.]
    bb3d = np.append(bb3d,aabb)
    obb = pcds_list[i].get_oriented_bounding_box()
    obb.color = [1.,1.,0]
    #bb3d = np.append(bb3d,obb)
  
  bb3d = np.append(bb3d, pcd)
  bb3d = np.append(bb3d, ground_pcd)

  o3d.visualization.draw_geometries(bb3d)

def pcds_from_hdbscan(pcd, hdbscan, ground_pts):
  '''
  '''
  points = np.asarray(pcd.points)
  max_label = hdbscan.labels_.max()
  n_clusters = max_label + 1
  y_pred = hdbscan.labels_.astype(int)
  container = []
  for i in range(n_clusters):
    m = np.array([])
    container.append(m)
  
  print('container length',len(container))

  if(len(y_pred) == len(points)):
    print('ourah')
  else:
    print(f'shit: len(y_pred): {len(y_pred)}, len(points): {len(points)}')

  print(points[:2])
  
  for i, elem in enumerate(points):
    'y_pred[i] az i-ik pont klaszter indexe'
    idx = y_pred[i]
    if(idx == -1): continue
    container[idx] = np.append(container[idx],points[i])

  ground_points = np.copy(ground_pts)
  ground_points = np.asarray(ground_points[:,:3])
  ground_pcd = o3d.geometry.PointCloud()
  ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
  ground_pcd.paint_uniform_color([0,1.,0])

  next_dict = getNextPathDict()
  pcds = []
  aabbs = []
  
  for elem in container:
    points = np.array(elem).reshape(-1,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0,0.5,0.5])
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [1.,0., 0.]
    geometries = [pcd, ground_pcd, aabb]
    o3d.visualization.draw_geometries(geometries)
    user_input = input("Enter the claster number to save: ")
    print(f'user input claster was: {user_input}')
    pcd = colorPcdByCluster(pcd,int(user_input))
  
    savePcdToNpy(user_input, next_dict, pcd)

    pcds.append(pcd)
    aabbs.append(aabb)
  
  pcds.append(ground_pcd)
  pcds = pcds + aabbs
  #geometries.append(aabbs)
  #geometries.append(ground_pcd)
  #geometries = [pcds, ground_pcd]
  o3d.visualization.draw_geometries(pcds)
    
  return container

def getNextPathDict():
  dict = {}
  files = os.listdir(dev_tosave_path)
  print(f'dev_tosave_path len folders: {len(files)}')

  for e in range(len(files)):
    i = 1
    print(f'getnextpathdict: path:  {dev_tosave_path + f"{e}/" + f"{e}_{i}.npy"}')
    while os.path.exists(dev_tosave_path + f"{e}/"+ f"{e}_{i}.npy"):
      i += 1
    dict[e] = i
    print(f'last elem in dict[{e}] = {dict[e]}')

  return dict

def savePcdToNpy(u_i, dict, pcd):
  '''
  save a pcd to a user determined input given class as .npy file
  '''
  print(f"user input {u_i}")
  print(f'dict at u_i: {dict[int(u_i)]}')
  #todo: frameid_cluster_idx.npy format
  np.save(dev_tosave_path + "/%s/" % u_i + f"{u_i}_{dict[int(u_i)]}.npy" , np.asarray(pcd.points))
  dict[int(u_i)] = dict[int(u_i)] + 1
  #return 

def colorPcdByCluster(pcd, cluster):
  colors = {}
  colors.update({0: [1., 0., 0.]})
  colors.update({1: [1., 1., 0.]})
  colors.update({2: [0., 0., 1.]})
  colors.update({3: [0, 1., 0.]})
  colors.update({4: [1., 0., 1.]})
  colors.update({5: [0., 1., 0.]})
  colors.update({6: [0. , 0. ,0.]})
  pcd.paint_uniform_color(colors[cluster])
  return pcd

  
def loadPcdFromNpy(cluster):
  #todo: frameid_cluster_idx.npy format
  data_path = dev_tosave_path + cluster
  points = np.load()


def bounding_box(velo):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  # Create bounding box: just z axis trimming
  bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [-2.5,2.5]]  # set the bounds

  #bounding box just the closest area of the car, around 10m
  #bounds = [[-10, 10], [-10, 10], [-2.5,2.5]]  # set the bounds
  bounding_box_points = list(itertools.product(*bounds))  # create limit points
  bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

  # Crop the point cloud using the bounding box:
  pcd_croped = pcd.crop(bounding_box)

  # Display the cropped point cloud:
  #o3d.visualization.draw_geometries([pcd_croped])
  points = np.asarray(o3d.utility.Vector3dVector(pcd_croped.points))
  #histogram(points,2)
  
  return points

def groud_Detection(velo):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
  geometries = [pcd,origin]

  '''
  x_max = data[:,0].max()
  x_min = data[:,0].min()
  y_max = data[:,1].max()
  y_min = data[:,1].min()
  z_max = data[:,2].max()
  z_min = data[:,2].min()
  '''

  x_max = max(pcd.points, key=lambda x: x[0])
  y_max = max(pcd.points, key=lambda x: x[1])
  z_max = max(pcd.points, key=lambda x: x[2])
  x_min = min(pcd.points, key=lambda x: x[0])
  y_min = min(pcd.points, key=lambda x: x[1])
  z_min = min(pcd.points, key=lambda x: x[2])
  print(f"{x_min[0]}:{x_max[0]}],[{y_min[1]}:{y_max[1]}],[{z_min[2]}:{z_max[2]}]")

  # Colors:
  RED = [1., 0., 0.]
  GREEN = [0., 1., 0.]
  BLUE = [0., 0., 1.]
  YELLOW = [1., 1., 0.]
  MAGENTA = [1., 0., 1.]
  CYAN = [0., 1., 1.]
  BLACK = [0. , 0. ,0.]

  positions = [x_max, y_max, z_max, x_min, y_min, z_min]
  colors = [RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN]
  for i in range(len(positions)):
   # Create a sphere mesh:
   sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
   # move to the point position:
   sphere.translate(np.asarray(positions[i]))
   # add color:
   sphere.paint_uniform_color(np.asarray(colors[i]))
   # compute normals for vertices or faces:
   sphere.compute_vertex_normals()
   # add to geometry list to display later:
   geometries.append(sphere)

  # Display min-max:
  #o3d.visualization.draw_geometries(geometries)

  # Define a threshold:
  THRESHOLD = 1.
  GREEN = [0., 1., 0.]

  # Number of points:
  n_points = points.shape[0]
  print('n_points', n_points)

  # Get the original points color to be updated:
  pcd_colors = np.zeros(shape=(n_points,3),dtype=float)
  print(pcd_colors.shape)

  for i in range(n_points):
    pcd_colors[i] = BLACK

  print('pcd color shape',pcd_colors.shape)

  ground_points = []
  not_ground_points = []

  # update color:
  for i in range(n_points):
    # if the current point is aground point:
    if pcd.points[i][2] <= z_min[2] + THRESHOLD:
        pcd_colors[i] = GREEN  # color it green
        #ground_points = np.append(ground_points, pcd.points[i])
        ground_points.append(pcd.points[i])
    #else: not_ground_points = np.append(not_ground_points, pcd.points[i])
    else: not_ground_points.append(pcd.points[i])

  #ground_points = np.asarray(ground_points).reshape((int(len(ground_points)/3),3))
  #not_ground_points = np.asarray(not_ground_points).reshape((int(len(not_ground_points)/3),3))

  ground_points = np.asarray(ground_points).reshape((int(len(ground_points)),3))
  not_ground_points = np.asarray(not_ground_points).reshape((int(len(not_ground_points)),3))
        
  print('ground points shape', ground_points.shape)
  print('not_ground_point shape', not_ground_points.shape)

  pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

  # Display:
  o3d.visualization.draw_geometries([pcd, origin])

  return ground_points, not_ground_points


'''
def my_plot(left_cam, velo, label_data, calib_data):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  #print('pcdpoints.shape', pcd.points.shape)

  geometries = [pcd]

  bb3d = labelTo3DBoundingBoxinPcd(label_data,calib_data)
  #print('bb3d.shape',bb3d.shape)
  print('bb3d',bb3d[0].shape)
  print('bb3d[0]', bb3d[0])
  print('bb3d[0][:1]', bb3d[0][0,:])

  print('pcd_x_max',max(pcd.points, key=lambda x: x[0]))
  print('pcd_y_max',max(pcd.points, key=lambda x: x[1]))
  print('pcd_z_max',max(pcd.points, key=lambda x: x[2]))
  print('pcd_x_min',min(pcd.points, key=lambda x: x[0]))
  print('pcd_y_min',min(pcd.points, key=lambda x: x[1]))
  print('pcd_z_min',min(pcd.points, key=lambda x: x[2]))

  bb3d_T = bb3d[0].T
  print('bb3D.T.shape', bb3d_T.shape)
  x_max = max(bb3d_T, key=lambda x: x[0])
  y_max = max(bb3d_T, key=lambda x: x[1])
  z_max = max(bb3d_T, key=lambda x: x[2])
  x_min = min(bb3d_T, key=lambda x: x[0])
  y_min = min(bb3d_T, key=lambda x: x[1])
  z_min = min(bb3d_T, key=lambda x: x[2])
  
  
  x_max = max(bb3d[0][0,:])
  y_max = max(bb3d[0][1,:])
  z_max = max(bb3d[0][2,:])
  x_min = min(bb3d[0][0,:])
  y_min = min(bb3d[0][1,:])
  z_min = min(bb3d[0][2,:])
  

  RED = [1., 0., 0.]
  GREEN = [0., 1., 0.]
  BLUE = [0., 0., 1.]
  YELLOW = [1., 1., 0.]
  MAGENTA = [1., 0., 1.]
  CYAN = [0., 1., 1.]

  positions = [x_max, y_max, z_max, x_min, y_min, z_min]
  colors = [RED, GREEN, BLUE, MAGENTA, YELLOW, CYAN]
  for i in range(len(positions)):
   # Create a sphere mesh:
   sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.)
   # move to the point position:
   sphere.translate(np.asarray(positions[i]))
   # add color:
   sphere.paint_uniform_color(np.asarray(colors[i]))
   # compute normals for vertices or faces:
   #sphere.compute_vertex_normals()
   # add to geometry list to display later:
   geometries.append(sphere)

  print('x_max', x_max)

  o3d.visualization.draw_geometries(geometries) 
'''


def main (frame='0000000000.bin'):

  filter_pipeline_timings()
  quit()

  '''
  velo = loadVeloFiles(frame)
  points = np.asarray(velo[:,:3])
  t_delta, len_optics_points = optics(points)
  print(f'{t_delta}, {len_optics_points}')
  '''

  #print(label_data)
  
  #original_plot(left_cam, velo, label_data, calib_data)
  #print(velo.shape)
  #print(velo[:2,:3])

  #my_plot(left_cam, velo, label_data, calib_data)

  #down_sample(velo)

  ind_rad, ind_stat =  outliers(points)
 
  #pointcloud_density(velo)
  #histogram(velo,2)
  croped = bounding_box(ind_stat)

  ground_pts, not_ground_pts = groud_Detection(croped)

  #optics(not_ground_pts)

  #dbscan(not_ground_pts)
  _pcd, _hdbscan = hdbscan(not_ground_pts)
  
  pcds_list = label_hdbscan(_pcd, _hdbscan)

  container = pcds_from_hdbscan(_pcd, _hdbscan, ground_pts)
  print(type(container[0]))

  print(container[:1])
  
  show_all(_pcd, ground_pts, pcds_list)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame', type=str,
                      default='0000000000.bin',
                      help='frame name with extension')
  FLAGS, unparsed = parser.parse_known_args()
  #print ('FLAGS', FLAGS)
  main(frame=FLAGS.frame)
