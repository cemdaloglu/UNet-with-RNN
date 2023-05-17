import napari
from readWrite import readArray, loadNpz
import numpy as np
from PIL import Image
import glob
import os
from moviepy.editor import ImageSequenceClip


def createVideo(files_list, rotation_coor, output_filepath, imgNo, fileName, threshold = True):

    bound_cube = np.zeros((64, 64, 64*len(files_list)))
    for i in range(1, len(files_list)+1):
        bound_cube[0, 0, :] = 1
        bound_cube[:, 0, 0] = 1
        bound_cube[0, :, 0] = 1
        bound_cube[63, 63, :] = 1
        bound_cube[:, 63, 64*i-1] = 1
        bound_cube[63, :, 64*i-1] = 1
        bound_cube[0, 63, :] = 1
        bound_cube[:, 0, 64*i-1] = 1
        bound_cube[0, :, 64*i-1] = 1
        bound_cube[63, 0, :] = 1
        bound_cube[:, 63, 0] = 1
        bound_cube[63, :, 0] = 1
    
    for i in range(imgNo):
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(bound_cube, rotate=rotation_coor, colormap='gray', blending='additive', rendering='mip')
        viewer.axes.visible = True
        for j in range(len(files_list)):
            tmp = readArray( files_list[j] )
            vol = np.zeros((tmp.shape[0], 64, 64, 64*(j+1)))
            vol[:, :, :, 64*j:64*(j+1)] = tmp
            if j == len(files_list)-1 and threshold:
                vol = np.where(vol > 1e-5, 1, 0)
            viewer.add_image(vol[i-imgNo+vol.shape[0], :, :, :], rotate=rotation_coor, colormap='gray', blending='additive', rendering='mip')
        if i < 10:
            viewer.screenshot(output_filepath + f"0{i}.png")
        else:
            viewer.screenshot(output_filepath + f"{i}.png")

    # Create the frames
    frames = []
    imgs = glob.glob(output_filepath + "*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    imgs = sorted(imgs)
    clip = ImageSequenceClip(list(imgs), fps=0.01)
    clip.write_gif(output_filepath + f'{fileName}.gif')

def delete_PNG(output_filepath):
    imgs = glob.glob(output_filepath + "*.png")
    for file in imgs:
        os.remove(file)

def openVolume(vol_list):

    viewer = napari.Viewer(ndisplay=3)

    for vol in vol_list:
        viewer.add_image(vol, colormap='gray', blending='additive', rendering='mip')

    napari.run()


if __name__ == '__main__':

    total_img_no = 39998
    total_pred_no = 18
    idx = 1

    gt_dataFilePath = f"C:/data/2022-11-21_guidewire_noBin_40000x16x64_p64_SO1.0E+06_cem/Results/ex_{39998 - idx}_voxSTL_guidewire_64x64x64x16.npz"
    pred1FilePath = f"C:/src/Predict_Results_Exp0_LongVideo/predict_{18 - idx}_voxSTL_guidewire_64x64x64x10.raw"
    pred2FilePath = f"C:/src/Predict_Results_Exp3_LongVideo/predict_{18 - idx}_voxSTL_guidewire_64x64x64x13.raw"

    saveVideoFilePath = "C:/src/output_videos_rnn/"
    videoName = f"bestLSTM_Base_GT_{39998-idx}"

    gt =  loadNpz(gt_dataFilePath)
    pred1 = readArray(pred1FilePath)
    pred2 = readArray(pred2FilePath)
    
    vol_vox2 = np.where(gt > 1e-5, 1, 0)

    openVolume([pred1, pred2, gt])

    delete_PNG(saveVideoFilePath)
    createVideo([pred1FilePath, pred2FilePath, gt_dataFilePath], rotation_coor=(3, 3, 0), output_filepath=saveVideoFilePath, imgNo=10, filename=videoName)