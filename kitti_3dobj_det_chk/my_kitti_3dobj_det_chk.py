#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:38:17 2018

The KITTI is one of the well known benchmarks for 3D Object detection. Working 
on this dataset requires some understanding of what the different files and their
contnts are. In this piece we use  4 different types of files used from the 
KITTI 3D Objection Detection dataset as follows to do some basic manipulation 
and sanity checks to get basic underdstanding. 

 camera2 image (.png), 
 camera2 label label (.txt),
 calibration (.txt), 
 velodyne point cloud (.bin),

Codes to project 3D data from  camera co-ordinate and velodyne coordinate to  
camera image.  The goal is to see if the  data along with appropriate geometry 
matrices are handled correctly.  2 different types of images are generated - 
camera2 image and bird's eye view of point cloud.  



@author: sg

Refs :
  1. Vision meets Robotics: The KITTI Dataset - http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
  2. 3D Object Detection Evaluation 2017 - http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
  3. Download left color images of object data set (12 GB) - http://www.cvlibs.net/download.php?file=data_object_image_2.zip
  4. Download Velodyne point clouds, if you want to use laser information (29 GB) - http://www.cvlibs.net/download.php?file=data_object_velodyne.zip
  5. Download camera calibration matrices of object data set (16 MB) - http://www.cvlibs.net/download.php?file=data_object_calib.zip
  6. Download training labels of object data set (5 MB) - http://www.cvlibs.net/download.php?file=data_object_label_2.zip
  7. Download object development kit (1 MB) (including 3D object detection and bird's eye view evaluation code) - http://kitti.is.tue.mpg.de/kitti/devkit_object.zip
  
"""

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


#basedir = 'C:/data/kitti/data_small/training' # windows
basedir = ".\data" # windows
#basedir = '../data.2/object/training' # *nix
#basedir = 'data' # *nix
left_cam_rgb= 'image_2'
label = 'label_2'
velodyne = 'velodyne'
calib = 'calib'

dev_velodyne_path = source = "C:/Users/Havasi/source/repos/resources/image_02_for_dev/velodyne/"
dev_tosave_path = "./temp_data/"


def loadKittiFiles (frame) :
  '''
  Load KITTI image (.png), calibration (.txt), velodyne (.bin), and label (.txt),  files
  corresponding to a shot.

  Args:
    frame :  name of the shot , which will be appended to externsions to load
                the appropriate file.
  '''
  # load image file 
  fn = basedir+ left_cam_rgb + frame+'.png'
  fn = os.path.join(basedir, left_cam_rgb, frame+'.png')
  #print("os.path:" + os.path.dirname)
  print("fn: " + fn)
  left_cam = Image.open(fn).convert ('RGB')
  
  # load velodyne file 
  fn = basedir+ velodyne + frame+'.bin'
  fn = os.path.join(basedir, velodyne, frame+'.bin')
  fn = dev_velodyne_path + '0000000000.bin'
  velo = np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
  
  # load calibration file
  fn = basedir+ calib + frame+'.txt'
  fn = os.path.join(basedir, calib, frame+'.txt')
  calib_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if ':' in line :
        key, value = line.split(':', 1)
        calib_data[key] = np.array([float(x) for x in value.split()])
  
  # load label file
  fn = basedir+ label + frame+'.txt'
  fn = os.path.join(basedir, label, frame+'.txt')
  label_data = {}
  with open (fn, 'r') as f :
    for line in f.readlines():
      if len(line) > 3:
        key, value = line.split(' ', 1)
        #print ('key', key, 'value', value)
        if key in label_data.keys() :
          label_data[key].append([float(x) for x in value.split()] )
        else:
          label_data[key] =[[float(x) for x in value.split()]]

  for key in label_data.keys():
    label_data[key] = np.array( label_data[key])
    
  return left_cam, velo, label_data, calib_data



def computeBox3D(label, P):
  '''
  takes an object label and a projection matrix (P) and projects the 3D
  bounding box into the image plane.
  
  (Adapted from devkit_object/matlab/computeBox3D.m)
  
  Args:
    label -  object label list or array
  '''
  w = label[7]
  h = label[8]
  l = label[9]
  x = label[10]
  y = label[11]
  z = label[12]
  ry = label[13]
  
  # compute rotational matrix around yaw axis
  R = np.array([ [+cos(ry), 0, +sin(ry)],
                 [0, 1,               0],
                 [-sin(ry), 0, +cos(ry)] ] )

  # 3D bounding box corners

  x_corners = [0, l, l, l, l, 0, 0, 0] # -l/2
  y_corners = [0, 0, h, h, 0, 0, h, h] # -h
  z_corners = [0, 0, 0, w, w, w, w, 0] # --w/2
  
  x_corners += -l/2
  y_corners += -h
  z_corners += -w/2
  
  
  # bounding box in object co-ordinate
  corners_3D = np.array([x_corners, y_corners, z_corners])
  #print ( 'corners_3d', corners_3D.shape, corners_3D)
  
  # rotate 
  corners_3D = R.dot(corners_3D)
  #print ( 'corners_3d', corners_3D.shape, corners_3D)
  
  #translate 
  corners_3D += np.array([x, y, z]).reshape((3,1))
  #print ( 'corners_3d', corners_3D)
  
  #print(corners_3D.shape)
  #print(f"shape[-1]: {corners_3D.shape[-1]}")
  
  corners_3D_1 = np.vstack((corners_3D,np.ones((corners_3D.shape[-1]))))
  corners_2D = P.dot (corners_3D_1)
  corners_2D = corners_2D/corners_2D[2]
  
  # edges, lines 3d/2d bounding box in vertex index 
  edges = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,0], [0,5], [1,4], [2,7], [3, 6]]
  lines = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,0], [0,5], [5, 4], [4, 1], [1,2], [2,7], [7,6], [6,3]]
  bb3d_lines_verts_idx = [0,1,2,3,4,5,6,7,0,5,4,1,2,7,6,3]
  
  bb2d_lines_verts = corners_2D[:,bb3d_lines_verts_idx] # 
   
  return corners_2D[:2], corners_3D, bb2d_lines_verts[:2]
  
  
  
 
def labelToBoundingBox(ax, labeld, calibd):
  '''
  Draw 2D and 3D bpunding boxes.  
  
  Each label  file contains the following ( copied from devkit_object/matlab/readLabels.m)
  #  % extract label, truncation, occlusion
  #  lbl = C{1}(o);                   % for converting: cell -> string
  #  objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
  #  objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
  #  objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
  #  objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])
  #
  #  % extract 2D bounding box in 0-based coordinates
  #  objects(o).x1 = C{5}(o); % left   -> in pixel
  #  objects(o).y1 = C{6}(o); % top
  #  objects(o).x2 = C{7}(o); % right
  #  objects(o).y2 = C{8}(o); % bottom
  #
  #  % extract 3D bounding box information
  #  objects(o).h    = C{9} (o); % box width    -> in object coordinate
  #  objects(o).w    = C{10}(o); % box height
  #  objects(o).l    = C{11}(o); % box length
  #  objects(o).t(1) = C{12}(o); % location (x) -> in camera coordinate 
  #  objects(o).t(2) = C{13}(o); % location (y)
  #  objects(o).t(3) = C{14}(o); % location (z)
  #  objects(o).ry   = C{15}(o); % yaw angle  -> rotation aroun the y/vetical axis
  '''
  
  # Velodyne to/from referenece camera (0) matrix
  Tr_velo_to_cam = np.zeros((4,4))
  Tr_velo_to_cam[3,3] = 1
  Tr_velo_to_cam[:3,:4] = calibd['Tr_velo_to_cam'].reshape(3,4)
  #print ('Tr_velo_to_cam', Tr_velo_to_cam)
  
  Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
  #print ('Tr_cam_to_velo', Tr_cam_to_velo)
  
  # 
  R0_rect = np.zeros ((4,4))
  R0_rect[:3,:3] = calibd['R0_rect'].reshape(3,3)
  R0_rect[3,3] = 1
  #print ('R0_rect', R0_rect)
  P2_rect = calibd['P2'].reshape(3,4)
  print('P2_rect', P2_rect)
  
  bb3d = []
  bb2d = []
  
  print('labeld.keys', labeld.keys())
  for key in labeld.keys ():
    
    color = 'white'
    if key == 'Car':
      color = 'red'
    elif key == 'Pedestrian':
      color = 'pink'
    elif key == 'Cyclist':
      color = 'purple'
    elif key == 'DontCare':
      color = 'white'
    
    print('labeld[key].shape', labeld[key].shape)
    for o in range( labeld[key].shape[0]):
      
      #2D
      left   = labeld[key][o][3]
      bottom = labeld[key][o][4]
      width  = labeld[key][o][5]- labeld[key][o][3]
      height = labeld[key][o][6]- labeld[key][o][4]
    
      
      p = patches.Rectangle(
        (left, bottom), width, height, fill=False, edgecolor=color, linewidth=1)
      ax.add_patch(p)
      
      xc = (labeld[key][o][5]+labeld[key][o][3])/2
      yc = (labeld[key][o][6]+labeld[key][o][4])/2
      bb2d.append([xc,yc])
      
      #3D
      w3d = labeld[key][o][7]
      h3d = labeld[key][o][8]
      l3d = labeld[key][o][9]
      x3d = labeld[key][o][10]
      y3d = labeld[key][o][11]
      z3d = labeld[key][o][12]
      yaw3d = labeld[key][o][13]
      
   
      if key != 'DontCare' :
        
        corners_2D, corners_3D, paths_2D = computeBox3D(labeld[key][o], P2_rect)
        print('path2d.shape', paths_2D.shape)
        print('path2d.type', paths_2D.dtype)
        verts = paths_2D.T # corners_2D.T
        print('verts.shape', verts.shape)
        #print('verts', verts)
        codes = [Path.LINETO]*verts.shape[0]
        #print('codes', codes)
        codes[0] = Path.MOVETO
        pth  = Path (verts, codes)
        p = patches.PathPatch( pth, fill=False, color='purple', linewidth=2)
        ax.add_patch(p)
        
  # a sanity test point in velodyne co-ordinate to check  camera2 imaging plane projection
  testp = [ 11.3, -2.95, -1.0]
  bb3d.append(testp)
  
  xnd = np.array(testp+[1.0])
  #print ('bb3d xnd velodyne   ', xnd)
  #xpnd = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(xnd)))
  xpnd = Tr_velo_to_cam.dot(xnd)
  #print ('bb3d xpnd cam0      ', xpnd)
  xpnd = R0_rect.dot(xpnd)
  #print ('bb3d xpnd rect cam0 ', xpnd)
  xpnd = P2_rect.dot(xpnd)
  #print ('bb3d xpnd cam2 image', xpnd)
  #print ('bb3d xpnd cam2 image', xpnd/xpnd[2])
  
  p = patches.Circle( (xpnd[0]/xpnd[2], xpnd[1]/xpnd[2]), fill=False, radius=3, color='red', linewidth=2)
  ax.add_patch(p)
  
  return np.array(bb2d), np.array(bb3d)

#labelto3Dboundingoxinpcd----------------------------------------------------
def labelTo3DBoundingBoxinPcd(labeld, calibd):
  # Velodyne to/from referenece camera (0) matrix
  Tr_velo_to_cam = np.zeros((4,4))
  Tr_velo_to_cam[3,3] = 1
  Tr_velo_to_cam[:3,:4] = calibd['Tr_velo_to_cam'].reshape(3,4)
  #print ('Tr_velo_to_cam', Tr_velo_to_cam)
  
  Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
  #print ('Tr_cam_to_velo', Tr_cam_to_velo)
  
  # 
  R0_rect = np.zeros ((4,4))
  R0_rect[:3,:3] = calibd['R0_rect'].reshape(3,3)
  R0_rect[3,3] = 1
  #print ('R0_rect', R0_rect)
  P2_rect = calibd['P2'].reshape(3,4)
  print('P2_rect', P2_rect)

  bb3d = []
  
  for key in labeld.keys ():
    color = 'white'
    if key == 'Car':
      color = 'red'
    elif key == 'Pedestrian':
      color = 'pink'
    elif key == 'Cyclist':
      color = 'purple'
    elif key == 'DontCare':
      color = 'white'
    
    for o in range( labeld[key].shape[0]):

      if key != 'DontCare' :
        
        corners_2D, corners_3D, paths_2D = computeBox3D(labeld[key][o], P2_rect)
        bb3d.append(corners_3D)
        '''
        verts = paths_2D.T # corners_2D.T
        codes = [Path.LINETO]*verts.shape[0]
        codes[0] = Path.MOVETO
        pth  = Path (verts, codes)
        p = patches.PathPatch( pth, fill=False, color='purple', linewidth=2)
        ax.add_patch(p)
        '''

  return bb3d



#PointCloud to bird eye --------------------------------------------------
def pointCloudToBirdsEyeView(ax2, velo, bb3d):
  ax2.set_xlim (-10,10)
  ax2.set_ylim (-5,35)
  hmax = velo[:,2].max()
  hmin = velo[:,2].min()
  hmean = velo[:, 2].mean()
  hmeadian = np.median ( velo[:, 2] )
  hstd = np.std(velo[:, 2])
  #print ('scalledh', hmax, hmean, hmeadian, hmin, hstd, scalledh.shape, scalledh[:10])
  norm = colors.Normalize(hmean-2*hstd, hmean+2*hstd, clip=True)
  sc2= ax2.scatter(-velo[:,1],
             velo[:,0],
             s = 1,
             c=velo[:,2],
             cmap = 'viridis',
             norm=norm,
             marker = ".",
             )
  ax2.scatter(-bb3d[:,1],
             bb3d[:,0],
             c='red')
  ax2.set_facecolor('xkcd:grey')
  plt.colorbar(sc2)

def down_sample(velo):
  points = np.asarray(velo[:,:3])
  #print(points.shape)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  random_pcd = pcd.random_down_sample(sampling_ratio=0.05)
  uniform_pcd = pcd.uniform_down_sample(every_k_points=10)
  voxel_pcd = pcd.voxel_down_sample(voxel_size=0.1)

  o3d.visualization.draw_geometries([pcd])
  o3d.visualization.draw_geometries([random_pcd])
  o3d.visualization.draw_geometries([uniform_pcd])
  o3d.visualization.draw_geometries([voxel_pcd])

def outliers(velo):
  points = np.asarray(velo[:,:3])
  #print(points.shape)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
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
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def filter_pipeline(velo):
  #shaping
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  #rho
  rho = pointcloud_density(velo)
  #print(rho)

  #voxel down sample
  voxel_pcd = pcd.voxel_down_sample(voxel_size=rho*2)
  o3d.visualization.draw_geometries([voxel_pcd])

  #statistical outlier removal
  pcd_stat, ind_stat = voxel_pcd.remove_statistical_outlier(nb_neighbors=16,std_ratio=1.5)
  display_inlier_outlier(voxel_pcd,ind_stat) 

  #z axis histogram
  histogram(velo, 2)
  #ground detection


  #crop bounding box 
  

#DBSCAN---------------------------------------------------------------------
def dbscan(velo):
  pointcloud_density(velo)

  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.4, min_points=16, print_progress=True))

  max_label = labels.max()
  print(f"point cloud has {max_label +1 } clusters")
  colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
  colors[labels < 0 ] = 0
  pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
  o3d.visualization.draw_geometries([pcd])

#OPTICS-----------------------------------------------------------------
def optics(velo):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  #downsample
  #uniform_pcd = pcd.uniform_down_sample(every_k_points=10)
  random_pcd = pcd.random_down_sample(sampling_ratio=0.1)
  points = np.asarray(random_pcd.points)

  #clust = OPTICS(min_samples=16, xi=0.05, min_cluster_size=0.05)
  clust = OPTICS(min_samples=16, xi=0.03)

  print("before fit")
  clust.fit(points)

  print("after fit")

  #cluster_optics_dbscan
  labels_050 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=0.5,
  )
  labels_200 = cluster_optics_dbscan(
    reachability=clust.reachability_,
    core_distances=clust.core_distances_,
    ordering=clust.ordering_,
    eps=2,
  )

  space = np.arange(len(points[:,0]))
  reachability = clust.reachability_[clust.ordering_]
  print(reachability.shape)
  labels = clust.labels_[clust.ordering_]
  print(labels.shape)
  _ = plt.hist(labels, bins='auto')
  plt.show()

  plt.figure(figsize=(10, 7))
  G = gridspec.GridSpec(2, 3)
  ax1 = plt.subplot(G[0, :])
  ax2 = plt.subplot(G[1, 0])
  ax3 = plt.subplot(G[1, 1])
  ax4 = plt.subplot(G[1, 2])

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

  # DBSCAN at 2.
  colors = ["g.", "m.", "y.", "c."]
  for klass, color in zip(range(0, 4), colors):
    Xk = points[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
  ax4.plot(points[labels_200 == -1, 0], points[labels_200 == -1, 1], "k+", alpha=0.1)
  ax4.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")

  plt.tight_layout()
  plt.show()

#HDBSCAN----------------------------------------------------------------------
def hdbscan(velo):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  #downsample
  #uniform_pcd = pcd.uniform_down_sample(every_k_points=10)
  pcd = pcd.random_down_sample(sampling_ratio=0.5)
  points = np.asarray(pcd.points)


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
  print("end fit")
  print('fit time: ' + str(t1-t0))

  y_pred = hdbscan.labels_.astype(int)
  n_cluster = len(set(y_pred))
  max_label = hdbscan.labels_.max()
  print(f"point cloud has {max_label +1 } clusters")

  colors = plt.get_cmap("tab20")(y_pred / (n_cluster if n_cluster > 0 else 1))
  colors[y_pred < 0] = 0

  pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

  o3d.visualization.draw_geometries([pcd])
  return pcd, hdbscan

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
  ground_pcd = o3d.geometry.PointCloud()
  ground_pcd.points = o3d.utility.Vector3dVector(points)

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



def pointcloud_density(velo):
  X = np.asarray(velo[:,:3])
  print(X.shape)
  nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
  distances, indices = nbrs.kneighbors(X)
  print(distances)
  rho = np.average(distances[:,1])
  print(rho)
  return rho
  


def histogram(velo, idx):
  points = np.asarray(velo[:,:3])
  _ = plt.hist(points[:,idx], bins='auto')
  plt.show()

def bounding_box(velo):
  points = np.asarray(velo[:,:3])
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)

  # Create bounding box: just z axis trimming
  #bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [-2.5,2.5]]  # set the bounds

  #bounding box just the closest area of the car, around 10m
  bounds = [[-10, 10], [-10, 10], [-2.5,2.5]]  # set the bounds
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

#show the original main function part
def original_plot(left_cam, velo, label_data, calib_data):
  """
  Completes the plots 
  """
  
  f = plt.figure(figsize=plt.figaspect(0.5))
  
  # show the left camera image 
  ax = f.add_subplot(3,1,1,)
  ax.imshow(left_cam)
  
  
  bb2d, bb3d = labelToBoundingBox(ax, label_data, calib_data)
  #print ('bb3d', bb3d)
  
  # point cloud to bird's eye view scatter plot
  ax2 = f.add_subplot(3,1,2, )#projection="3d" )
  pointCloudToBirdsEyeView(ax2, velo, bb3d)

  plt.show()

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
  
  '''
  x_max = max(bb3d[0][0,:])
  y_max = max(bb3d[0][1,:])
  z_max = max(bb3d[0][2,:])
  x_min = min(bb3d[0][0,:])
  y_min = min(bb3d[0][1,:])
  z_min = min(bb3d[0][2,:])
  '''

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

  '''
  bounding_box_points = [[x_min, y_min, z_min],[x_max,y_max,z_max]]
  
  bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object

  # Crop the point cloud using the bounding box:
  pcd_croped = pcd.crop(bounding_box)
  '''
  # Display the cropped point cloud:
  #o3d.visualization.draw_geometries([pcd_croped]) 
  o3d.visualization.draw_geometries(geometries) 


def main (frame='000008'):

  left_cam, velo, label_data, calib_data = loadKittiFiles(frame)

  #print(label_data)
  
  #original_plot(left_cam, velo, label_data, calib_data)
  #print(velo.shape)
  #print(velo[:2,:3])

  #my_plot(left_cam, velo, label_data, calib_data)

  #down_sample(velo)

  ind_rad, ind_stat =  outliers(velo)
 
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
                      default='000008',
                      help='frame name without extension')
  FLAGS, unparsed = parser.parse_known_args()
  #print ('FLAGS', FLAGS)
  main(frame=FLAGS.frame)
