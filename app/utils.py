def describe_dict(d: dict, max_sample_size=3) -> dict:
    """
    Recursively describes the structure, types, and provides a small sample of a dictionary.
    
    Args:
    d (dict): The dictionary to describe.
    max_sample_size (int): Maximum number of sample items for lists and nested structures.
    
    Returns:
    dict: A dictionary with keys as the original keys and values as their type descriptions and samples.
    """
    def describe(value):
        # Handle nested dictionaries
        if isinstance(value, dict):
            return {key: describe(val) for key, val in value.items()}
        
        # Handle lists with type tracking and sampling
        elif isinstance(value, list):
            if not value:
                return "list (empty)"
            
            # Check if all list elements are of the same type
            list_types = list(set(type(item).__name__ for item in value))
            
            # Prepare sample
            sample = value[:max_sample_size]
            
            if len(list_types) == 1:
                return {
                    "type": f"list of {list_types[0]}",
                    "sample": sample,
                    "total_length": len(value)
                }
            else:
                return {
                    "type": f"list with mixed types: {list_types}",
                    "sample": sample,
                    "total_length": len(value)
                }
        
        # Handle other types
        else:
            return {
                "type": type(value).__name__,
                "sample": value
            }
    
    # Describe the entire dictionary
    return {key: describe(val) for key, val in d.items()}