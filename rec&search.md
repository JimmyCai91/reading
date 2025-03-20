# Evaluation

## Offline

### Ref.

- [common metrics to evaluate rec. systems](https://flowthytensor.medium.com/some-metrics-to-evaluate-recommendation-systems-9e0cf0c8b6cf)

### AUC
<div align="center"><img src="figures/rec&search_auc.webp" width=""></div>

``` SQL
-- SQL Calc AUC

WITH auc_info AS (
    select channel, uv, action_uv, g_cnt, action_g_cnt, 
        (action_ry - 0.5 * action_n1 * (action_n1 + 1)) / action_n0 / action_n1 as auc
    from (  
        select channel, 
            count(distinct user_id) as uv, 
            count(distinct case when label = 1 then user_id end) as action_uv, 
            count(1) as g_cnt, -- 'g' is short for group
            count(case when label = 1 then 1 end) as action_g_cnt, 
            sum(if (label = 0, 1, 0)) as action_n0, 
            sum(if (label = 1, 1, 0)) as action_n1,
            sum(if (label = 1, rk, 0)) as action_ry
        from (
            select channel, user_id, search_id, product_id, label, 
                row_number() over (partition by channel order by rank_score ASC) AS rk
            from  base_info
        )
        group by 
            channel
    )
)

-- SQL Calc UAUC
WITH rank_info AS (
    SELECT user_id, product_id, search_id, label, rank_score, 
        row_number() over (partition by user_id order by rank_score ASC) AS u_rk, 
    FROM base_info
),

pick_uauc_info AS (
    select channel, 
        sum(pv) as action_pv, 
        count(inner_count) as user_count, 
        sum(inner_count) as count, 
        sum(action_auc * inner_count) / sum(inner_count) AS uauc
    from (
        select channel, user_id, pv, inner_count, sum_label, 
            (action_ry - 0.5 * action_n1 * (action_n1 + 1)) / action_n0 / action_n1 as action_auc
        from (
            select channel, user_id, 1 as pv, 
                count(1) as inner_count, 
                sum(label) as sum_label, 
                sum(if (label = 0, 1, 0)) as action_n0,
                sum(if (label = 1, 1, 0)) as action_n1,
                sum(if (label = 1, u_rk, 0)) as action_ry
            from rank_info 
            group by user_id, channel
        )
        where sum_label > 0 
        and sum_label <> inner_count
    )
    group by channel
)
```

### Precision @ K
<div align="center"><img src="figures/rec&search_precisionatk.webp" width=""></div>

```python 
def precision_at_k(results, k):
    """
    Parameters:
    - results: A list of binary values indicating whether the item at each rank is relevant (1) or not relevant (0).
    - k: The rank position up to which precision is calculated.

    Returns:
    - Precision at k: The proportion of relevant items in the top k results.
    """

    if k > len(results):
        raise ValueError("k should be less than or equal to the number of results")

    top_k_results = results[:k]
    relevant_items = sum(top_k_results)
    precision = relevant_items / k
    return precision
```

### Recall @ K
<div align="center"><img src="figures/rec&search_recallatk.webp" width=""></div>

```python 
def recall_at_k(results, k):
    """
    Parameters:
    - results: A list of binary values indicating whether the item at each rank is relevant (1) or not relevant (0).
    - k: The rank position up to which recall is calculated.

    Returns:
    - Precision at k: The proportion of relevant items in the top k results.
    """

    if k > len(results):
        raise ValueError("k should be less than or equal to the number of results")

    top_k_results = results[:k]
    relevant_items_in_top_k = sum(top_k_results)
    total_relevant_items = sum(results)

    if total_relevant_items == 0:
        return 0.0

    recall = relevant_items_in_top_k / total_relevant_items
    return recall
```


### Average Precision @ K
<div align="center"><img src="figures/rec&search_averageprecisionatk.webp" width=""></div>

```python 
def average_precision_at_k(results,k):
    """
    Parameters:
    - results: A list of binary values indicating whether the item at each rank is relevant (1) or not relevant (0)
    - k: The rank position up to which average precision is calculated 

    Returns:
    - Average Precision at k: The average of precision at each rank up to k
    """
    if k > len(results):
        raise ValueError("k should be less than or equal to the number of results")

    precisions = []
    relevant_count = 0

    for i in range(1, k+1):
        if results[i-1] == 1:
            relevant_count += 1
            precisions.append(relevant_count / i)
    
    if not precisions:
        return 0.0

    average_precision = sum(precisions) / len(precisions)
    return average_precision 
```

### NDCG @ K
<div align="center"><img src="figures/rec&search_ndcgatk.webp" width=""></div>

```python
import numpy as np

def dcg_at_k(results, k):
    """
    Parameters:
    - results: A list of binary values indicating whether the item at each rank is relevant (1) or not relevant (0)
    - k: The rank position up to which DCG is calculated 

    Returns:
    - DCG at k
    """
    results = np.asfarray(results)[:k]
    if results.size:
        return results[0] + np.sum(results[1:] / np.log2(np.arange(2, results.size + 1)))
    return 0.0

def ndcg_at_k(results, k):
    """
    Parameters:
    - results: A list of binary values indicating whether the item at each rank is relevant (1) or not relevant (0)
    - k: The rank position up to which NDCG is calculated 

    Returns:
    - NDCG at k
    """
    if k > len(results):
        raise ValueError("k should be less than or equal to the number of results")

    # DCG calculation
    dcg_max = dcg_at_k(sorted(results, reverse=True), k)
    dcg_actual = dcg_at_k(results, k)

    # NDCG calculation
    if not dcg_max:
        return 0.0
    return dcg_actual / dcg_max

```