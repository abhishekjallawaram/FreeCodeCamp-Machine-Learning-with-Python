## Introduction

Welcome to Google Colaboratory, a cloud-hosted version of Jupyter Notebook. This environment combines documentation and code, allowing for an interactive learning and development experience. If new to Jupyter Notebooks, consider watching this [3-minute introduction](https://www.youtube.com/watch?v=inN8seMm7UI).

## Challenge Overview

You are tasked with developing a book recommendation algorithm leveraging the K-Nearest Neighbors (KNN) technique. Utilize the [Book-Crossings dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/), comprising over 1.1 million book ratings ranging from 1 to 10, contributed by approximately 90,000 users for 270,000 books.

### Objectives

1. **Data Preparation**: Import and cleanse the dataset, ensuring only books with at least 100 ratings and users with at least 200 ratings are considered. This step enhances the statistical relevance of the data.

2. **Model Development**: Employ `NearestNeighbors` from `sklearn.neighbors` to construct a model that identifies books similar to a specified title based on user ratings.

3. **Recommendation Function**: Implement `get_recommends`, a function that accepts a book title and returns five similar books along with their proximity scores.

### Example

Input:

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```

Expected Output:
```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```


