# TopoLayer
![TopoLayer](https://github.com/Zechao-Guan/TopoLayer/blob/main/Topolayer.pdf)

The illustration depicts the pipeline for extracting topological features within a point cloud and the neural network architecture of TopoLayer. This pipeline consists of two main steps: (1) The generation of persistence diagrams derived from the filtration of Vietoris-Rips complexes applied to a point cloud; (2) The generation of topology features by TopoLayer. For various dimensional persistence diagrams, TopoLayer can employ PPDTF, PDTF, or a combination of both methods to vectorize these persistence diagrams. Consequently, a weight function denoted as $\omega\left(\cdot\right)$ is applied to each resulting vector, and a permutation-invariant operation represented as **op** is utilized to extract topology features. Subsequently, these topology features are concatenated to obtain a global topology feature. This process offers a range of parameter options, providing flexibility and adaptability.

# The Generation of Persistence Diagrams
![image](https://github.com/Zechao-Guan/TopoLayer/blob/main/Topolayer.pdf)

## Result
By introducing TopoLayer topological features extracted, various tasks of point cloud analysis, such as classification and part segmentation tasks, can be enhanced.

Integration of TopoLayer, without architectural modifications, significantly improves established models such as PointMLP and PointNet++. 
- For classification on ModelNet40, the class mean accuracy of PointMLP notably improves from 91.3\% to 91.8\%. Additionally, PointNet++ achieves a remarkable gain of 2.7\%, elevating its performance from 90.7\% to 93.4\%. 
- For part segmentation on ShapeNetPart, PointMLP achieves a new state-of-the-art performance with 85.1\% classification mean IoU, while PointNet++ secures a significant 0.9\% increase, boosting its classification mean IoU from 81.9\% to 82.8\%.
- Moreover, TopoLayer demonstrates its effectiveness in confronting loss of geometric information.

Here are files of the experiment results: [Result](https://drive.google.com/drive/folders/1iFS2vJjwxr5lBL0OXoVkO3KaFpZ04v_n?usp=sharing)

## Useage
### Classification ModelNet40
```python
cd ModelNet40
# train pointMLP
python train_TopoPointMLP_tailversion.py
# train pointNet++
python train_TopoPointNet2_headversion.py
```

### Classification ModelNet40 with Noise
```python
cd ModelNet40_Noise
# train pointMLP
python train_TopoPointMLP_tailversion.py
# train pointNet++
python train_TopoPointNet2_headversion.py
```

### Part segmentation
```python
cd Partsegment
# train pointMLP
python train_TopoPointMLP_tailversion.py
# train pointNet++
python train_TopoPointNet2_headversion.py
```
