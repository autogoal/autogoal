#   This file implements hierarchical divisive clustering for autogoal. 

import math
import numpy as np
import random

class MyCluster():
    def __init__(self, id, elements ):
        self.id = id
        self.elements = elements

class Division():
    
    def __init__(self, x_distance: str, c_distance: str, stop: str):
        self.x_distance = self.x_distances[x_distance]
        self.c_distance = self.c_distances[c_distance]
        self.stop = self.stops[stop]

    def c_distance_single(self, ci, cj):
        x1, x2 = None, None
        d = np.inf
        for xl in ci:
            for xm in cj:
                if (not (xl == xm).all()):
                    temp_d = self.x_distance(self, xl, xm)
                    if  temp_d < d:
                        d = temp_d
                        x1 = xl
                        x2 = xm      
        return d

    def c_distance_complete(self, ci, cj):
        x1, x2 = None, None
        d = 0
        for xl in ci:
            for xm in cj:
                if (not (xl == xm).all()):
                    temp_d = self.x_distance(self, xl, xm)
                    if  temp_d > d:
                        d = temp_d
                        x1 = xl
                        x2 = xm 
        return d

    def x_distance_euc(self, xl, xm):
        temp = 0
        for i in range(xl.shape[0]):
                temp = temp + (xm[i] - xl[i])**2
        return math.sqrt(temp)

    def stop_dummy(self, old_cluster, new_cluster):
        return random.randrange(0,5) > 3

    def sw(self, clusters):
        sw = 0
        for c in range(len(clusters)):
            for x in clusters[c].elements:
                for y in clusters[c].elements:
                    if (x != y).all():
                        sw = sw + self.x_distance(self, x, y)
        return sw/2
    
    def sb(self, clusters):
        sb = 0
        for c1 in clusters:
            for c2 in clusters:
                if ((c1 != c2)):
                    sb += self.c_distance(self,c1.elements, c2.elements)
        return sb/2

    def nw(self, clusters):
        nw = 0
        for c in clusters:
            nw += (c.elements.shape[0] * c.elements.shape[0] - 1) / 2
        return nw

    def nb(self, clusters):
        n = 0
        for c in clusters:
            n += c.elements.shape[0]
        return ((n * (n - 1)) / 2) - self.nw(clusters)

    def stop_frey(self, old_cluster, new_cluster):
        old_sw = self.sw(old_cluster)
        old_sb = self.sb(old_cluster)
        
        new_sw = self.sw(new_cluster)
        new_sb = self.sb(new_cluster)

        old_nw = self.nw(old_cluster)
        old_nb = self.nb(old_cluster)
        
        new_nw = self.nw(new_cluster)
        new_nb = self.nb(new_cluster)

        temp1 = new_nw
        if temp1 == 0:
            return np.inf

        temp = ((new_sw / new_nw) - (old_sw) / (old_nw) )
        if temp == 0:
            return np.inf
        return (((new_sb / new_nb) - (old_sb / old_nb) / temp )) < 1
    
    x_distances={
        'euclidean': x_distance_euc,
    }
    
    c_distances={
        'single': c_distance_single,
        'complete': c_distance_complete
    }
    stops={
        'dummy': stop_dummy,
        'frey': stop_frey,

    }
    
    # The run fucntion implements the algotithm for divisive clustering technique
    def run(self, input: np.ndarray) -> np.ndarray:
        #paso 1
        
        m = input.transpose()
        # d_to_item = np.zeros((m.shape[0], m.shape[0]))
        # for x in range(m.shape[0]):
        
        id = 0

        #creating the firs cluster with all the items
        cluster = MyCluster(0, m)
        id = id + 1

        #list containning all clusters that must be analized in following iterations
        clusters_list = [cluster]

        #list containning the current cluster division
        current_clusters = [cluster]

        while(clusters_list):
            cluster = clusters_list.pop(0)
            print('entrando con el cluster:')
            print(cluster.elements)
            
            while(cluster.elements.shape != () and cluster.elements.shape[0] > 1):
                d_elem_cluster = 0
                
                #  Represents the candidat to become the first element in the new clsuter
                candidat = None

                #   Kips a copy of all the items but the candidat 
                elements_copy = []

                #   Compute the distance from every item i to the rest of the cluster
                for element in cluster.elements:   
                    temp_d = self.c_distance(self,[element], cluster.elements)
                    if temp_d > d_elem_cluster:
                        d_elem_cluster = temp_d
                        if candidat is not None:  
                            elements_copy.append(candidat)
                        candidat = element.copy()
                    else: 
                        elements_copy.append(element)
                
                elements_copy = np.array(elements_copy)

                #   Create the new cluster with the element with the biggest distance computed
                candidat_elements = np.array([candidat])
                new_cluster = MyCluster(id, candidat_elements)
                id = id + 1
                
                d_to_old_cluster=0
                d_to_new_cluster=0

                #   Compute wish elements will join the new cluster
                elements_copy_copy = elements_copy.copy()
                if (len(elements_copy) > 1):    
                    for x in range(len(elements_copy_copy)):
                        elem = elements_copy_copy[x]

                        d_to_old_cluster = self.c_distance(self, [elem], elements_copy)
                        d_to_new_cluster = self.c_distance(self, [elem], new_cluster.elements)

                        temp_dif = d_to_old_cluster - d_to_new_cluster

                        if (temp_dif > 0):
                            #   Join to the new cluster
                            elem_array = np.array([elem])
                            new_cluster.elements = np.append(new_cluster.elements, elem_array, axis=0)
                            elem_list = elem.tolist()
                            elements_copy_list = elements_copy.tolist()
                            elements_copy_list.remove(elem_list)
                            elements_copy = np.array(elements_copy_list)
                        
                
                old_clusters = current_clusters.copy()
                current_clusters.remove(cluster)
                cluster = MyCluster(id, elements_copy)
                id += 1
                current_clusters += [new_cluster] + [cluster]

                #   Evaluate if we have obtanined the best division
                if  self.stop(self, old_clusters, current_clusters ):
                    result = np.zeros((m.shape[0], 1))
                    for i in range(m.shape[0]):
                        for c in old_clusters:
                            for x in c.elements:
                                if (m[i] == x).all():
                                    result[i] = c.id
                    return result

                clusters_list.append(new_cluster)  

        
#   Example
division = Division('euclidean','single', 'frey')
a = [[1,2,3,4,5], [1, 4, 10, 500, 503]]
b = np.ndarray((2,5))
for x in range(len(a)):
    for y in range(len(a[x])):
        b[x,y] = a[x][y]

c = division.run(b)

print(c)


# from autogoal.ml import AutoML
# from autogoal.contrib import find_classes

# # Probemos con HAHA
# from autogoal.datasets import haha

# # Cargando los datos
# X_train, y_train, X_test, y_test = haha.load()


# # Creando la instancia de AutoML con nuestra clase
# automl = AutoML(
#     input=MatrixContinuousDense,  # **tipos de entrada**
#     output=MatrixContinuousDense,  # **tipo de salida**
#     # Agregando nuestra clase y todo el resto de algortimos de AutoGOAL
#     registry=[Division] + find_classes(),
# )

# # Ahora sencillamente tenemos que ejecutar AutoML y ya nuestro algoritmo aparecerá en algunos pipelines.
# # Debemos tener en cuenta que esto no garantiza qeu aparezca en el mejor pipeline encontrado, sino que se conectará
# # con el resto de los algoritmo como si fuera nativo de AutoGOAL.

# automl.fit(X_train, y_train)

# score = automl.score(X_test, y_test)
# print(score)