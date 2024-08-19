def maybe_flatten_the_list(lst):
    flat_list = []
    
    def flatten(sublist):
        for item in sublist:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)
                
    flatten(lst)
    return flat_list