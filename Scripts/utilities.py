import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from matplotlib.gridspec import GridSpec

from pandas.plotting import parallel_coordinates

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import cohen_kappa_score

from scipy.spatial.distance import pdist

from scipy.spatial.distance import squareform

import plotly.express as xp

import matplotlib.pyplot as plt


plt.rcParams["axes.grid"] = True
markers = ['v', 'p', 'd', '<', 'P', '>', '<', '+', '.', '+', 'o', '>', '>', 'p', '>', 'v', '^', 'v', '*', 'D']
colors = ['darkslategrey', 'mediumorchid', 'mediumblue', 'mediumspringgreen',
    'darkseagreen', 'darkolivegreen', 'mediumseagreen', 'darkcyan',
    'darkviolet', 'mediumvioletred', 'darksalmon', 'darkgoldenrod',
    'darkgreen', 'darkgrey', 'mediumaquamarine', 'darkblue',
    'darkslateblue', 'darkorange', 'darkgray', 'darkorchid']
    

def get_dissect_profiles(raw_dissect_output, threshold=0.04):

  # Remove results with "-" label as they correspond to background (pixles with no class)
  raw_data = raw_dissect_output[raw_dissect_output["label"] != "-"]
  # Remove untis with class matching less than 0.04 iou (as par Dissect authors)
  raw_data = raw_data[raw_data["iou"]>threshold]
  # Normalize Profile Values
  # unique concepts
  # 66 object
  u_objects, objects_counts = np.unique(raw_data[raw_data["category"]=="object"]["label"], return_counts=True)
  # 52 part
  u_parts, parts_counts = np.unique(raw_data[raw_data["category"]=="part"]["label"], return_counts=True)
  # 15 material
  u_materials, materials_counts = np.unique(raw_data[raw_data["category"]=="material"]["label"], return_counts=True)
  # 8 color
  u_colors, colors_counts = np.unique(raw_data[raw_data["category"]=="color"]["label"], return_counts=True)

  # Count number of unites matched to each concept
  aggregation = raw_data.groupby(["name","label", "category"], as_index=False).agg({"iou": ["count"]})
  aggregation.columns = ["name", "label", "category", "count"]

  # Rename labels based on category
  aggregation.loc[ aggregation["category"]=="object" , "label"] = "o_" + aggregation.loc[ aggregation["category"]=="object" , "label"] 
  aggregation.loc[ aggregation["category"]=="part" , "label"] = "p_" + aggregation.loc[ aggregation["category"]=="part" , "label"] 
  aggregation.loc[ aggregation["category"]=="material" , "label"] = "m_" + aggregation.loc[ aggregation["category"]=="material" , "label"] 
  aggregation.loc[ aggregation["category"]=="color" , "label"] = "c_" + aggregation.loc[ aggregation["category"]=="color" , "label"] 

  # Normalize counts by the total number of found matches for each category
  aggregation["count"]  = aggregation["count"] / (aggregation["category"]=="object").map(lambda x: len(u_objects) if x else 1)
  aggregation["count"]  = aggregation["count"] / (aggregation["category"]=="material").map(lambda x: len(u_materials) if x else 1)
  aggregation["count"]  = aggregation["count"] / (aggregation["category"]=="part").map(lambda x: len(u_parts) if x else 1)
  aggregation["count"]  = aggregation["count"] / (aggregation["category"]=="color").map(lambda x: len(u_colors) if x else 1)

  # Reshape data as a vector of found concepts for each model
  aggregation = aggregation.pivot(index='name', columns='label')["count"]
  dissect_profiles = pd.DataFrame(aggregation.values, columns = aggregation.keys().tolist(), index=aggregation.index)
  dissect_profiles.reset_index()
  dissect_profiles.fillna(0, inplace=True)

  return dissect_profiles

def get_abstracted_dissect_profile(raw_data, threshold=0.04):
  # Generate Abstracted Profile
  raw_data = raw_data[raw_data["label"] != "-"]
  raw_data = raw_data[raw_data["iou"]>threshold]
  dissect_abstract_profiles = raw_data[["name", "category", "label"]].copy()
  dissect_abstract_profiles = dissect_abstract_profiles.pivot(columns="category", values="label")#.replace('.*R.*', '1', regex=True)
  dissect_abstract_profiles = pd.concat([raw_data[["name"]],dissect_abstract_profiles], axis=1)
  dissect_abstract_profiles = dissect_abstract_profiles.groupby("name").agg(["count", "nunique"])
  dissect_abstract_profiles.columns = [' '.join(col).strip() for col in dissect_abstract_profiles.columns]
  dissect_abstract_profiles.columns = ["all_"+c.split(" ")[0] if "count" in c else "unique_"+c.split(" ")[0]
                              for c in dissect_abstract_profiles.columns]
  dissect_abstract_profiles = dissect_abstract_profiles[["all_object", "all_part", "all_material", "all_color",
                "unique_object","unique_part","unique_material","unique_color",]]
  return dissect_abstract_profiles

def elbow(dissect_profiles):
  # Remove models that doesn't have performance data (Custom SwAVs) or have a very weak dissect profile (SimCLR+Random)
  remove = ["SimCLRV1(R)", "SimCLRV2(R)", "Random(R)", "SwAV200(R)", "SwAV400(R)", "SwAV200bs256(R)", "SwAV400_2x244(R)"]
  f_s_data = dissect_profiles[~dissect_profiles.index.isin(remove)].iloc[:,1:]
  inertias = []

  assignments = []
  K = range(1,11)

  for k in K:
      kmeanModel = KMeans(n_clusters=k)
      kmeanModel.fit(f_s_data)
      if k==3:
        assignments = kmeanModel.labels_
      inertias.append(kmeanModel.inertia_)

  plt.plot(K, inertias, 'bx-')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.title('The Elbow Method to Decide Number of Clusters')
  plt.xticks(K, K)
  plt.show()
  return 

def compute_embedding(dissect_profiles):
  # Remove models that doesn't have performance data (Custom SwAVs) or have a very weak dissect profile (SimCLR+Random)
  remove = ["SimCLRV1(R)", "SimCLRV2(R)", "Random(R)", "SwAV200(R)", "SwAV400(R)", "SwAV200bs256(R)", "SwAV400_2x244(R)"]
  f_s_data = dissect_profiles[~dissect_profiles.index.isin(remove)].iloc[:,1:]

  pca = PCA(n_components=3)
  components = pca.fit_transform(f_s_data)
  pca1 = [ v for (v, _, _) in components]
  pca2 = [ v for (_, v, _) in components]
  pca3 = [ v for (_, _, v) in components]
  components = pd.DataFrame(components, columns=["pca1", "pca2", "pca3"])
  components["model"] = [l for l in f_s_data.index ]
  return components, pca

def plot_embedding(dissect_profiles, components):
  # Remove models that doesn't have performance data (Custom SwAVs) or have a very weak dissect profile (SimCLR+Random)
  remove = ["SimCLRV1(R)", "SimCLRV2(R)", "Random(R)", "SwAV200(R)", "SwAV400(R)", "SwAV200bs256(R)", "SwAV400_2x244(R)"]
  f_s_data = dissect_profiles[~dissect_profiles.index.isin(remove)].iloc[:,1:]

  pca1 = components.pca1
  pca2 = components.pca2
  pca3 = components.pca3

  fig = plt.figure(figsize=(9, 4))
  gs = GridSpec(1,3, fig)

  # Plot first and second componenets scatter plot
  ax1 = fig.add_subplot(gs[0,:2])
  ax1.set_ylabel("Second component")
  ax1.set_xlabel("First component")
  for x, y, l, marker, color in zip(pca1, pca2, [l for l in f_s_data.index ], markers, colors):
    ax1.scatter(x, y, label=l, marker=marker, color=color, s=75)
  ax1.axvspan(-3, 6,  alpha=0.075, color="red", ymin=-1, ymax=0.45)
  ax1.axhspan(-2, 3,  alpha=0.075, color="green", xmin=-1, xmax=0.42)
  ax1.annotate( "Low Features Redundancy", (1, -0.6), xytext=(2, -0.5), bbox={"boxstyle":"round", "fc":"1"}, size=8,)
  ax1.annotate( "High Features Redundancy",  (1, -0.6), xytext=(-2, 0.1), bbox={"boxstyle":"round", "fc":"1"}, size=8, rotation=90)
  ax1.set_xlim((-2.7, 5.2))
  ax1.set_ylim((-1.5, 2.2))

  # Plot third component scatter plot
  ax2 =  fig.add_subplot(gs[0,2])
  ax2.set_xlabel("Third component")
  for x, y, l, marker, color in zip(pca3, pca2, [l for l in f_s_data.index], markers, colors):
    ax2.scatter(x, y, label=l, marker=marker, color=color, s=75)
  ax2.set_yticklabels([])
  ax2.set_xticks(range(-2, 2, 1))

  fig.tight_layout(pad=0.2)
  plt.legend(loc="upper right", bbox_to_anchor=(1.8, 0.98))
  plt.suptitle("Visual Features Embedding of Models \n Based On Concepts Found By Dissect", position=(0.65,1.11))
  plt.show()
  return 

def plot_embedding_coefficients(dissect_profiles, pca):
  # Remove models that doesn't have performance data (Custom SwAVs) or have a very weak dissect profile (SimCLR+Random)
  remove = ["SimCLRV1(R)", "SimCLRV2(R)", "Random(R)", "SwAV200(R)", "SwAV400(R)", "SwAV200bs256(R)", "SwAV400_2x244(R)"]
  f_s_data = dissect_profiles[~dissect_profiles.index.isin(remove)].iloc[:,1:]

  # Componenets Coefficients as Dataframe
  comps = pd.DataFrame([pca.components_[0], pca.components_[1], pca.components_[2]],
                      columns= [c.replace("o_","").replace("c_","").replace("m_","").replace("p_","") for c in f_s_data.columns])
  comps["class"] = ["First Principal Component", "Second Principal Componnet", "Third Principal Componnet"]

  # Make the plot 
  plt.figure(figsize=(25,5))
  parallel_coordinates(comps, class_column="class", color=["red", "green", "blue"])
  plt.xticks(rotation=90)
  # Add shades
  plt.axvspan(0, 8, alpha=0.1, color="red")
  plt.axvspan(8, 23,  alpha=0.1, color="blue" )
  plt.axvspan(23, 92, alpha=0.1, color="green")
  plt.axvspan(92, 144, alpha=0.1, color="magenta")

  plt.title("The Detailed Composition Of The First Three Principal Components")
  plt.show()
  return  


def plot_embedding_performance(components, performance_data, learning_task="dt_ImageNet", with_legend=True):
  
  pca1 = components.pca1
  pca2 = components.pca2
  pca3 = components.pca3

  components_perf = components.merge(performance_data[["model", learning_task]], on="model")
  components_perf = components_perf[components_perf["model"]!="SeLaV1(R)"]

  clf = KNeighborsRegressor(5, weights='distance')
  clf.fit(components_perf[["pca1", "pca2"]].values, components_perf[learning_task].values)

  x= np.arange(-3, 6, 0.2)
  y=  np.arange(-3, 6, 0.2)
  X, Y = np.meshgrid(x, y)
  Z = clf.predict( np.stack([X.flatten(), Y.flatten()], axis=1) )
  Z = Z.reshape(len(x), len(y))
  plt.contourf(x, y, Z,  cmap="Reds")
  plt.colorbar()

  for x, y, l, marker, color in zip(pca1, pca2, [l for l in components_perf["model"] ], markers, colors):
    plt.scatter(x, y, label=l, marker=marker, color=color)

  if with_legend:
    plt.legend(loc="upper right", bbox_to_anchor=(1.8,1.02))
  plt.title("Semantic Space of models based on concepts found by Dissect ")
  plt.xlabel("pca.1")
  plt.ylabel("pca.2")
  plt.xlim((-3,5.5))
  plt.ylim((-1.3,2.8))

  plt.show()
  return

def correlate_component_abstract_profile(dissect_abstract_profiles, components):
  abstract_profile_columns = dissect_abstract_profiles.columns
  components_dissect = components.merge(dissect_abstract_profiles.reset_index(),
                                        left_on="model", right_on= "name")
  pca1_cor = components_dissect[abstract_profile_columns].corrwith(components_dissect["pca1"])
  pca2_cor = components_dissect[abstract_profile_columns].corrwith(components_dissect["pca2"])
  pca3_cor = components_dissect[abstract_profile_columns].corrwith(components_dissect["pca3"])

  x = np.arange(len(abstract_profile_columns))
  width = 0.3
  fig = plt.figure(figsize=(14,6))
  plt.bar(x - width, pca1_cor, width, label='First Principal Component')
  plt.bar(x , pca2_cor, width, label='Second Principal Component')
  plt.bar(x + width, pca3_cor, width, label='Third Principal Component')

  plt.axvline(3.5)

  plt.title("How the three main axes of variability correlates with abstracted Dissect profile")
  plt.xticks(x, labels=[l.replace("all_","All\n").replace("unique_", "Unique\n")
  for l in abstract_profile_columns], rotation=0)
  plt.legend()

  plt.show()
  return

def plot_abstracted_dissect_profiles(dissect_abstract_profiles, aggregation):
  if not aggregation.lower() in ["unique", "all"]:
    print("aggregation parameter should be in [\"unique\", \"all\"]")
  dissect_data = dissect_abstract_profiles[[c for c in dissect_abstract_profiles if aggregation in c]].copy().reset_index()
  dissect_data[aggregation+"_sum"] = dissect_data.sum(axis=1)
  dissect_data.sort_values(by=aggregation+"_sum", ascending=False, inplace=True)

  width= 0.5
  fig = plt.figure(figsize=(10,5))
  x = [m.replace("(R)", "") for m in dissect_data["name"]]

  objects = np.array(dissect_data[aggregation+"_object"])
  parts = np.array(dissect_data[aggregation+"_part"])
  materials = np.array(dissect_data[aggregation+"_material"])
  colors = np.array(dissect_data[aggregation+"_color"])

  plt.bar(x, objects, width=width, label="Objects")
  plt.bar(x, parts, width=width, label="Objects Parts", bottom=objects)
  plt.bar(x, materials, width=width, label="Material Parts", bottom=objects+parts)
  plt.bar(x, colors, width=width, label="Color Parts", bottom=objects+parts+materials)

  plt.legend()
  plt.title(aggregation.capitalize() + " Concepts Found by Dissect")
  plt.xticks(rotation=75)
  plt.ylabel("# concepts")

  plt.show()
  return


def plot_embedding_performance_correlation(components, performance_data, task):
  if task.lower() == "many shot classification":
    anchor = "dt_"
  elif task.lower() == "detection":
    anchor = "_ap"
  elif task.lower() == "few shot classification":
    anchor = "shot"
  elif task.lower() == "surface normal estimation":
    anchor = "sne"
  else:
    raise Exception("task parameter should be in ['many shot classification', 'detection', \
                    'few shot classification', 'surface normal estimation']")
    
  plt.figure(figsize=(16,4))
  cols = [ c for c in performance_data.columns if anchor in c]
  components_perf = components.merge(performance_data, on="model")
  components_perf = components_perf[components_perf["model"]!="SeLaV1(R)"]

  width=.3
  plt.bar(np.arange(0,len(cols))-width, components_perf[cols].corrwith(components_perf["pca1"]), width, label="1st Component" )
  plt.bar(np.arange(0,len(cols)), components_perf[cols].corrwith(components_perf["pca2"]), width, label="2nd Component" )
  plt.bar(np.arange(0,len(cols))+width, components_perf[cols].corrwith(components_perf["pca3"]), width, label="3rd Component" )

  labels = [c.replace("dt_", "").replace("_shot", "") for c in cols]

  plt.xticks(range(0,len(cols)), labels= labels, rotation=70)
  plt.title(f"Correlation Of Principal Components With Performance On {task.title()}")
  plt.ylabel("Correlation")
  plt.legend()
  plt.show()
  return 

def plot_embedding_performance_correlation(components, performance_data, task):
  if task.lower() == "many shot classification":
    anchor = "dt_"
  elif task.lower() == "detection":
    anchor = "_ap"
  elif task.lower() == "few shot classification":
    anchor = "shot"
  elif task.lower() == "surface normal estimation":
    anchor = "sne"
  else:
    raise Exception("task parameter should be in ['many shot classification', 'detection', \
                    'few shot classification', 'surface normal estimation']")
    
  plt.figure(figsize=(16,4))
  cols = [ c for c in performance_data.columns if anchor in c]
  components_perf = components.merge(performance_data, on="model")
  components_perf = components_perf[components_perf["model"]!="SeLaV1(R)"]

  width=.3
  plt.bar(np.arange(0,len(cols))-width, components_perf[cols].corrwith(components_perf["pca1"]), width, label="1st Component" )
  plt.bar(np.arange(0,len(cols)), components_perf[cols].corrwith(components_perf["pca2"]), width, label="2nd Component" )
  plt.bar(np.arange(0,len(cols))+width, components_perf[cols].corrwith(components_perf["pca3"]), width, label="3rd Component" )

  labels = [c.replace("dt_", "").replace("_shot", "") for c in cols]

  plt.xticks(range(0,len(cols)), labels= labels, rotation=70)
  plt.title(f"Correlation Of Principal Components With Performance On {task.title()}")
  plt.ylabel("Correlation")
  plt.legend()
  plt.show()
  return

def plot_embedding_performance_scatter(components, performance_data, task_dataset, correlation = "pearson"):
  merged = components.merge(performance_data[["model", task_dataset]], on="model")
  if "dt_" in task_dataset:
    task = "many shot classification"
  elif "_ap"  in task_dataset:
    task = "detection"
  elif "shot" in task_dataset:
    task = "few shot classification"
  elif "sne" in task_dataset:
    task = "surfce normal estimation"
  fig = plt.figure(figsize= (20,5))
  for i in range(1,4):
    plt.subplot(130 + i)
    x = merged[f"pca{i}"]
    y = merged[task_dataset]
    for xp, yp, marker, color, label in zip(x, y, markers, colors, merged["model"]):
      plt.scatter(xp, yp, marker= marker, color=color, label=label, s= 75)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, b + m*x)
    plt.xlabel(f"Component {i}")
    plt.ylabel(f"Task Performance")
    plt.title(f"{correlation}: {x.corr(y,correlation):2.3}")
  plt.suptitle(f"Perfomance On Task {task.title()} ({task_dataset})")
  _ = plt.legend(loc = (1.02, 0.15))
  return

##############################
##########ENSEMBLES###########
##############################

def load_models_unify(folders):
  for folder in folders:
    folder_dataframes = [];
    for filename in os.listdir(folder):
      if not "csv" in filename:
        continue
      model = filename.split("_")[2]
      dataframe = pd.read_csv(os.path.join(folder, filename), index_col=0)
      dataframe["model"] = model
      folder_dataframes.append(dataframe)
    folder_dataframe = pd.concat(folder_dataframes)
    folder_dataframe.to_csv(folder.split("/")[-1]+".csv")
  return

def compute_similarity_score(dataframe, score_name):
  if score_name == "kappa":
    score_fn = cohen_kappa_score
  elif score_name == "agreement":
    score_fn = lambda y1, y2 : (y1 == y2).sum()/len(y1)*100
  elif score_name == "euclidean":
    score_fn = lambda y1, y2 : np.mean(np.linalg.norm(y1 - y2, axis=1))
  else:
    raise Exception
  models = dataframe.model.unique()
  output = pd.DataFrame([], columns=["model1", "model2", score_name])
  ground_truth = dataframe[dataframe.model == models[0]]["gt"]
  for model1 in models:
    pred1_proba = dataframe[dataframe.model == model1].iloc[:, :-2].values
    pred1 = pred1_proba.argmax(axis=1)
    for model2 in models:
      pred2_proba = dataframe[dataframe.model == model2].iloc[:,:-2].values
      pred2 = pred2_proba.argmax(axis=1)
      if score_name == "euclidean":
        score = score_fn(pred1_proba, pred2_proba)
      else:
        score = score_fn(pred1, pred2)
      output = output.append({"model1": model1, "model2": model2, score_name: score}, ignore_index=True)
  return output

def compute_ensemble(dataframe, approach = "weighted_soft"):
  models = dataframe.model.unique()
  output = pd.DataFrame([], columns=["model1", "model2", "perf1", "perf2", "perf_ensemble"])
  ground_truth = dataframe[dataframe.model == models[0]]["gt"]

  for model1 in models:
    pred1_proba = dataframe[dataframe.model == model1].iloc[:, :-2].values
    pred1 = pred1_proba.argmax(axis=1)
    perf1 = (pred1 == ground_truth).sum()/len(ground_truth)*100

    for model2 in models:
      pred2_proba = dataframe[dataframe.model == model2].iloc[:,:-2].values
      pred2 = pred2_proba.argmax(axis=1)
      perf2 = (pred2 == ground_truth).sum()/len(ground_truth)*100

      if approach == "weighted_soft":        
        diff_ratio = abs(perf1-perf2)/100
        w_pred1_proba = pred1_proba * (0.5 + ((-1)**(not perf1>=perf2))*diff_ratio)
        w_pred2_proba = pred2_proba * (0.5 + ((-1)**(not perf1<perf2))*diff_ratio)
        predBoth = (np.stack((w_pred1_proba, w_pred2_proba))).sum(axis=0).argmax(axis=1)
        perfBoth = (predBoth==ground_truth).sum()/len(ground_truth)*100

      output = output.append({"model1": model1, "model2": model2, "perf1": perf1, "perf2": perf2, "perf_ensemble": perfBoth}, ignore_index=True)
  return output

def pred_similarity_vs_distance(similarity, components, similarity_score_name):
  similarity = similarity[~((similarity.model1=="infomax")|(similarity.model2=="infomax"))]
  similarity_pairwise = similarity.pivot("model1", "model2")

  components.model = components.model.str.replace("\(R\)", "") \
                  .str.lower() \
                  .str.replace("deepcluster", "dc") \
                  .str.replace("v1", "") \
                  .str.replace("min", "max")
  components = components.set_index("model").loc[similarity_pairwise.index, :]
  distance_pairwise = pd.DataFrame(squareform(pdist(components, metric='cosine')),
                                  columns = components.index,
                                  index = components.index,
                                  )
  similarity_pairwise.values[np.tril_indices(similarity_pairwise.shape[0], 0)] = np.nan
  distance_pairwise.values[np.tril_indices(distance_pairwise.shape[0], 0)] = np.nan

  plt.scatter(similarity_pairwise, distance_pairwise)
  plt.xlabel(f"Similarity Score ({similarity_score_name})")
  plt.ylabel("Embedding Cosine Distance")
  _= plt.title("The Relation between models embedding\n distances and prediction similarity")
  return

def plot_detailed_ensemble_distance(ensemble_df, tag):
  ensemble_df = ensemble_df[~((ensemble_df.model1=="infomax")|(ensemble_df.model2=="infomax"))]
  ensemble_df_pairwise = ensemble_df.pivot("model1", "model2")
  ensemble_df["model1"] = ensemble_df.model1.str.upper()
  ensemble_df["model2"] = ensemble_df.model2.str.upper()
  
  reindex= ["DCV2", "SWAV", "SELAV2", "BYOL", "PCLV2",
          "SUPERVISED", "MOCOV2",  "PIRL", "INSDIS", "PCL", "MOCO", "SELA"]

  accs = ensemble_df.groupby(by="model1").min("perf1").loc[reindex,:]["perf1"]
  accs = np.diag(accs)

  ensemble_df["dif"] = (ensemble_df["perf_ensemble"] - ensemble_df[["perf1", "perf2"]].max(axis=1))
  ensemble_df = ensemble_df.pivot(index="model1",columns="model2", values="dif")
  ensemble_df = ensemble_df.loc[reindex, reindex]
  ensemble_df.fillna(0, inplace=True) 

  fig = plt.figure(figsize=(10,10))
  plt.matshow(ensemble_df.abs(), cmap="hot_r", fignum=1)


  for (i, j), z in np.ndenumerate(ensemble_df+accs):
      if i==j:
        plt.text(j, i, '{:2.3}'.format(z), ha='center', va='center')
      else:
        plt.text(j, i, '{:1.2}'.format(z), ha='center', va='center')

  plt.title(f"Dissect Semantic Profile Cosine Distance\nCells Contains Performance Difference After Ensemble\n{tag}", position=(0.5,1.15))
  plt.xticks(range(0, len(reindex)), labels = reindex, rotation=90)
  plt.yticks(range(0, len(reindex)), labels = reindex)
  plt.colorbar()
  plt.grid(False)
  plt.show()