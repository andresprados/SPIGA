

cfg_retinasort = {

    'retina': {
        'model_name': 'mobile0.25',
        'extra_features': ['landmarks'],
        'postreat': {
            'resize': 1.,
            'score_thr': 0.75,
            'top_k': 5000,
            'nms_thr': 0.4,
            'keep_top_k': 50}
        },

    'sort': {
        'max_age': 1,
        'min_hits': 3,
        'iou_threshold': 0.3,
    }
}

cfg_retinasort_res50 = {

    'retina': {
        'model_name': 'resnet50',
        'extra_features': ['landmarks'],
        'postreat': {
            'resize': 1.,
            'score_thr': 0.75,
            'top_k': 5000,
            'nms_thr': 0.4,
            'keep_top_k': 50}
        },

    'sort': {
        'max_age': 1,
        'min_hits': 3,
        'iou_threshold': 0.3,
    }
}

cfg_retinasort_cav3d = {

    'retina': {
        'model_name': 'resnet50',
        'extra_features': ['landmarks'],
        'postreat': {
            'resize': 1.,
            'score_thr': 0.95,
            'top_k': 5000,
            'nms_thr': 0.8,
            'keep_top_k': 50}
        },

    'sort': {
        'max_age': 90,
        'min_hits': 3,
        'iou_threshold': 0.3,
    }
}

cfg_retinasort_av16 = {

    'retina': {
        'model_name': 'resnet50',
        'extra_features': ['landmarks'],
        'postreat': {
            'resize': 1.,
            'score_thr': 0.75,
            'top_k': 5000,
            'nms_thr': 0.8,
            'keep_top_k': 50}
        },

    'sort': {
        'max_age': 90,
        'min_hits': 3,
        'iou_threshold': 0.3,
    }
}
