# 模型的选择和评价
## accuracy, precision and recall

### confusion matrix

| |predict true|predict false|
| --- | ---| --- |
|True| True P| False N|
|False| False P| True N|

- **accuracy = all_true/ all**
- **predict= TP/TP+FP**
- **recall = TP/TP+FN**

### F1 score and F-beta score
```f1 = 2 * precison * recall/(precision + recall) (harmonic mean)```

```f-beta =  (1+ beat^2) * precision * recall / ((beta^2 *precision)+recall)```

- **To give more weight to the Precision, we pick a Beta value in the interval 0 < Beta < 1**
- **To give more weight to the Recall, we pick a Beta Value in the interval 1 < Beta < +∞**

## K-fold and validation and learning curve
  
