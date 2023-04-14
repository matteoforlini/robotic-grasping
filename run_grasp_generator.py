from inference.grasp_generator import GraspGenerator
import numpy as np
import os

homedir = os.path.join(os.path.expanduser('~'), "Desktop\prova\grasp-comms")
grasp_request = os.path.join(homedir, "grasp_request.npy")

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id='046122251438',
        saved_model_path='..\cornell-randsplit-rgbd-grconvnet3-drop1-ch32\epoch_15_iou_0.97',
        visualize=True
    )
    np.save(grasp_request, 1)
    generator.load_model()
    generator.run()
