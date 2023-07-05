import numpy as np 


def rasterize(p2d,Rows,Columns,H,W):
    """
        p2d: 2D points
        Rows: number of rows in the image
        Columns: number of columns in the image
        H: height of the camera sensor
        W: width of the camera sensor
    """
    
    num_points = p2d.shape[0]

    vertical_ratio = Rows/H
    horizontal_ratio = Columns/W
    RastCanvas = np.zeros((num_points,2))
    for i in range(num_points):
        RastCanvas[i][0] = np.around(np.floor(p2d[i][0]*vertical_ratio+1))
        RastCanvas[i][1] = np.around(np.floor(p2d[i][1]*horizontal_ratio+1))
        
    # Roatate the image by 90 degrees
    P_rast = np.zeros((RastCanvas.shape[0], 2))
    P_rast[:, 0] = RastCanvas[:, 1]
    P_rast[:, 1] = RastCanvas[:, 0]
    return P_rast
