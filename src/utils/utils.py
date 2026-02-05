def clean_ground_truth(gt_text):
    if str(gt_text).lower() in ['nan', "couldn't find", "pas de nom"]:
        return set()
    return set([s.strip() for s in str(gt_text).lower().split(',') 
                if s.strip() and len(s.split()) >= 2])
