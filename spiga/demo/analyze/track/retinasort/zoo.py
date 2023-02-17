# My libs
import spiga.demo.analyze.track.retinasort.face_tracker as tr
import spiga.demo.analyze.track.retinasort.config as cfg_tr


def get_tracker(model_name):

    # MobileNet Backbone
    if model_name == 'RetinaSort':
        return tr.RetinaSortTracker()
    # ResNet50 Backbone
    if model_name == 'RetinaSort_Res50':
        return tr.RetinaSortTracker(cfg_tr.cfg_retinasort_res50)
    # Config CAV3D: https://ict.fbk.eu/units/speechtek/cav3d/
    if model_name == 'RetinaSort_cav3d':
        return tr.RetinaSortTracker(cfg_tr.cfg_retinasort_cav3d)
    # Config AV16: https://ict.fbk.eu/units/speechtek/cav3d/
    if model_name == 'RetinaSort_av16':
        return tr.RetinaSortTracker(cfg_tr.cfg_retinasort_av16)

    return None


