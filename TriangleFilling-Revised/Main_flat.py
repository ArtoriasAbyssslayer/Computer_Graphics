import time
from common_utils.load_vertices import load_binary_data
from common_utils.render import render

# from common_utils.render import render

def main():# Define Canvas Boundaries
    # load vetices,faces, and arbitrary image elements from binary 

    verts2d, vcolors, faces, depth = load_binary_data('./GHW_assets/cg-hw1/h1.npy')

    # set timer 
    start = time.time()
    print('Triangle Filling on canvas initialized!')


    shaded_canvas = render(verts2d,faces,vcolors,depth,'flat')

    # set timer
    end = time.time()
    print('Triangle Filling Flat Shading Finished! Elapsed Seconds:{:.2f}'.format(end - start))
    from common_utils.render import renderImageToFile
    renderImageToFile(shaded_canvas,save=True,filename='Flat_Shading_Res')


if __name__ == '__main__':
    main()