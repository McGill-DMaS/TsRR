import numpy as np

def tsrr(target, results, similarities, alpha=0.5, reduction='mean'):
    """
    Tie-Sensitive Reciprocal Rank (tsrr) function.

    This function computed the tie-sensitive reciprocal rank metric.
    It accepts a target label (or multiple target labels), 
    corresponding result labels, and similarity scores. The inputs may be provided 
    as Python lists or NumPy ndarrays.

    The function handles inputs as follows:
    
    - **Single Target:**
      - If `target` is not a list or NumPy ndarray (or is a 0-dim NumPy scalar), it is 
        treated as a single target.
      - In this case, both `results` and `similarities` must be provided as lists or 1D 
        NumPy arrays of equal length.
    
    - **Multiple Targets:**
      - If `target` is a list or a 1D NumPy ndarray, it is treated as containing multiple target labels.
      - In this scenario, both `results` and `similarities` must be provided as 2D lists or 2D NumPy arrays.
      - The number of sublists in `results` and `similarities` must match the number of target labels.
      - Each sublist in `results` must have the same length as the corresponding sublist in `similarities`.

    Parameters
    ----------
    target : any, list, or numpy.ndarray
        A single target label or a collection of target labels. If not provided as a list or 
        1D array (or if provided as a 0-dim NumPy scalar), it is treated as a single target.
    results : list or numpy.ndarray
        For a single target, a list or 1D array of result labels.
        For multiple targets, a 2D list or 2D array where each sublist contains result labels 
        corresponding to a target label.
    similarities : list or numpy.ndarray
        For a single target, a list or 1D array of similarity scores corresponding element-wise 
        to `results`.
        For multiple targets, a 2D list or 2D array where each sublist contains similarity scores 
        corresponding to the result labels of a target.
    alpha : float, optional
        A positive float controlling the sensitivity to ties in the TSRR computation. A higher value 
        makes the metric more sensitive to ties. Default is 0.5.
    reduction : str, optional
        Specifies how to aggregate the TSRR scores:
          - 'mean': Return the mean TSRR score over all targets.
          - 'none': Return an array of TSRR scores for each target.
        Default is 'mean'.

    Returns
    -------
    tsrr_score : float or list
        The computed tie-sensitive reciprocal rank score(s). If `reduction` is 'mean', a single float is returned 
        representing the average TSRR over all targets. If `reduction` is 'none', a 1D list of TSRR scores 
        is returned, one for each target.

    Raises
    ------
    ValueError
        If the input formats do not meet the expected criteria. For example:
          - For a single target, if `results` and `similarities` are not lists or 1D arrays,
            or if their lengths differ.
          - For multiple targets, if `results` and `similarities` are not 2D lists or 2D arrays,
            or if the number of sublists does not match the number of target labels, or if any 
            corresponding sublists differ in length.

    Examples
    --------
    Single target example:
    >>> tsrr("label1", ["label2", "label1", "label3"], [0.8, 0.9, 0.7])
    0.6666666666666666   # (computed as 1/(2^alpha) when alpha=0.5)

    Multiple targets example:
    >>> tsrr(["label1", "label2"],
    ...      [["label1", "labelA"], ["labelB", "label2"]],
    ...      [[0.9, 0.8], [0.85, 0.95]],
    ...      alpha=0.5, reduction='none')
    array([1.0, 0.5])  # Example computed values
    """

    if not isinstance(target, (list, np.ndarray)):
        # Validate that results and similarities are lists of the same length.
        if not (isinstance(results, list) and isinstance(similarities, list)):
            raise ValueError("For a single target, both 'results' and 'similarities' must be lists.")
        if len(results) != len(similarities):
            raise ValueError("For a single target, 'results' and 'similarities' must have equal lengths.")
        
        target = [target]
        results = [results]
        similarities = [similarities]

    elif isinstance(target, np.ndarray) and target.ndim == 0:
        # Single target: target is a NumPy scalar.
        if not (isinstance(results, (list, np.ndarray)) and isinstance(similarities, (list, np.ndarray))):
            raise ValueError("For a single target, both 'results' and 'similarities' must be lists or numpy arrays.")
        if isinstance(results, np.ndarray):
            results = results.tolist()
        if isinstance(similarities, np.ndarray):
            similarities = similarities.tolist()
        if len(results) != len(similarities):
            raise ValueError("For a single target, 'results' and 'similarities' must have equal lengths.")
        
        target = [target.item()]  # Convert scalar to a Python type.
        results = [results]
        similarities = [similarities]

    else:
        # Multiple targets: target is either a list or a non-scalar ndarray.
        if isinstance(target, np.ndarray):
            # For multiple targets, require target to be 1D.
            if target.ndim != 1:
                raise ValueError("For multiple targets, 'target' must be a 1D array.")
            target = target.tolist()
        
        # Validate that results is 2D.
        if isinstance(results, np.ndarray):
            if results.ndim != 2:
                raise ValueError("For multiple targets, 'results' must be a 2D array.")
            results = results.tolist()
        else:
            if not (isinstance(results, list) and all(isinstance(row, list) for row in results)):
                raise ValueError("For multiple targets, 'results' must be a 2D list or a 2D numpy array.")
        
        # Validate that similarities is 2D.
        if isinstance(similarities, np.ndarray):
            if similarities.ndim != 2:
                raise ValueError("For multiple targets, 'similarities' must be a 2D array.")
            similarities = similarities.tolist()
        else:
            if not (isinstance(similarities, list) and all(isinstance(row, list) for row in similarities)):
                raise ValueError("For multiple targets, 'similarities' must be a 2D list or a 2D numpy array.")
        
        # The number of sublists in results and similarities should match the number of target labels.
        if len(results) != len(target):
            raise ValueError("For multiple targets, the number of sublists in 'results' must match the number of target labels.")
        if len(similarities) != len(target):
            raise ValueError("For multiple targets, the number of sublists in 'similarities' must match the number of target labels.")
        
        # Ensure each sublist in results and similarities have the same length.
        for i, (res_row, sim_row) in enumerate(zip(results, similarities)):
            if len(res_row) != len(sim_row):
                raise ValueError(f"Mismatch in lengths of sublists at index {i} between 'results' and 'similarities'.")
    
    if not isinstance(results, (np.ndarray)):
        results = np.array(results)

    if not isinstance(similarities, (np.ndarray)):
        similarities = np.array(similarities)

    tsrr_values = []
    for i, label in enumerate(target):
        sorted_indices = np.argsort(similarities[i])[::-1]
        sorted_similarities = similarities[i, sorted_indices]
        sorted_results = results[i, sorted_indices]
        target_indices = np.where(sorted_results == label)[0]
        if target_indices.size > 0:
            first_index = target_indices[0]
            target_score = sorted_similarities[first_index]
        else:
            tsrr_values.append(0)
            continue

        r_pre = np.where(sorted_similarities > target_score)[0].shape[0]
        F_total = sorted_results.shape[0] - np.where(sorted_results == label)[0].shape[0]
        same_score_indices = np.where(sorted_similarities == target_score)[0]
        F_g = np.where(sorted_results[same_score_indices] != label)[0].shape[0]
        if F_total == 0:
            tsrr_values.append(1)
            continue
        tsrr_values.append((1 - ((np.log(1 + F_g) / np.log(1 + F_total)) ** (1/alpha))) * (1/(r_pre+1)))

    if reduction == 'mean':
        return np.mean(tsrr_values)
    
    return tsrr_values
    