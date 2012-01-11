# -*- python -*-

import math
import numpy as np
cimport numpy as np
import scipy.sparse as sparse
import turtle

cdef extern from "math.h":
    double floor(double)
    double round(double)

cdef extern from "polygon.h":
    int poly_clip(int N, double* x, double* y,
                  double xleft, double xright, double ytop, double ybottom,
                  double* workx, double* worky)
    double area(int N, double* px, double* py)

cdef tf(double x, double y, np.ndarray M):
    cdef np.ndarray[np.double_t, ndim=2] H = M
    cdef double xx, yy, zz

    xx = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    yy = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    zz = H[2, 0] * x + H[2, 1] * y + H[2, 2]

    xx /= zz
    yy /= zz

    return xx, yy

cdef tf_polygon(int N, double* xp, double* yp, np.ndarray H_arr):
    cdef int i
    for i in range(N):
        xp[i], yp[i] = tf(xp[i], yp[i], H_arr)

def poly_interp_op(int MM, int NN, np.ndarray[np.double_t, ndim=2] H,
                   int M, int N, search_win=7, construct_prior=False):
    """
    Construct a linear interpolation operator based on polygon overlap.

    Parameters
    ----------
    MM, NN : int
        Shape of the high-resolution source frame.
    H : (3, 3) ndarray
        Transformation matrix that warps the high-resolution image to
        the low-resolution image.
    M, N : int
        Shape of the low-resolution target frame.
    search_win : int
        Search window size.  Note TODO: this parameter should
        be automatically determined.

    Returns
    -------
    op : (M*N, MM*NN) sparse array
        Interpolation operator.

    """
    cdef double rx[5]
    cdef double ry[5]
    cdef double xleft, xright, ytop, ybottom
    cdef double a, S
    cdef double workx[9], worky[9]

    cdef list I = [], J = [], V = [], W = []

    cdef int m, n, wr, wc
    cdef int K, k, verts, hwin

    cdef int out_M = M * N
    cdef int out_N = MM * NN

    cdef np.ndarray[np.double_t, ndim=2] inv_tf_M = np.linalg.inv(H)

    cdef double mt, nt, ridx, cidx
    cdef int skip

    # For each element in the low-resolution source
    for m in range(M):
        for n in range(N):
            # Create pixel polygon
            #
            # The 0.5 offset is to generate a pixel around point (m, n)
            #
            xleft = n - 0.5
            xright = xleft + 1
            ybottom = m - 0.5
            ytop = ybottom + 1

            # Find position in high-resolution frame
            nt, mt = tf(n, m, inv_tf_M)
            nt = round(nt)
            mt = round(mt)

            # For 25 pixels in target vicinity
            K = 0
            hwin = (search_win - 1)/2
            skip = False
            for wr in range(-hwin, hwin):
                if skip:
                    break
                for wc in range(-hwin, hwin):
                    rx[0] = nt + wc - 0.5
                    rx[1] = rx[0] + 1
                    rx[2] = rx[1]
                    rx[3] = rx[0]
                    rx[4] = rx[0]

                    ry[0] = mt + wr - 0.5
                    ry[1] = ry[0]
                    ry[2] = ry[0] + 1
                    ry[3] = ry[2]
                    ry[4] = ry[0]

                    ridx = round(mt + wr - 0.5)
                    cidx = round(nt + wc - 0.5)

                    # Ameliorate edge effects by skipping out of the loop
                    # whenever we hit the sides
                    if ridx < 0 or ridx > (MM - 1) or \
                       cidx < 0 or cidx > (NN - 1):
                        skip = True
                        break

                    # Project back to LR frame
                    tf_polygon(5, rx, ry, H)

                    verts = poly_clip(5, rx, ry,
                                      xleft, xright, ytop, ybottom,
                                      workx, worky)
                    a = area(verts, workx, worky)

                    I.append(m*N + n)
                    J.append((int)(ridx * NN + cidx))
                    V.append(a)
                    
                    if construct_prior:
                        

                    K += 1

            S = 0
            for k in range(K):
                S += V[-k]

            for k in range(K):
                if S > 1e-6:
                    V[-k] /= S

    return sparse.coo_matrix((V, (I, J)), shape=(out_M, out_N)).tocsr()


def drawPolygon(px, py, n):
    zoom = 8.
    turtle.penup()
    turtle.setposition(px[0]*zoom, py[0]*zoom)
    turtle.pendown()
    for i in range(1,n):
        turtle.setposition(px[i]*zoom, py[i]*zoom)
    turtle.penup()
        
def poly_interp_op_yaw(int MM, int NN,
                   int M, int N, float yaw, float zoom, search_win=7):
    """
    Construct a linear interpolation operator based on polygon overlap.

    Parameters
    ----------
    MM, NN : int
        Shape of the high-resolution source frame.
    H : (3, 3) ndarray
        Transformation matrix that warps the high-resolution image to
        the low-resolution image.
    M, N : int
        Shape of the low-resolution target frame.
    search_win : int
        Search window size.  Note TODO: this parameter should
        be automatically determined.

    Returns
    -------
    op : (M*N, MM*NN) sparse array
        Interpolation operator.

    """
    cdef double rx[5]
    cdef double ry[5]
    cdef double px[5]
    cdef double py[5]
    
    _rx = np.zeros(5)
    _ry = np.zeros(5)
    _px = np.zeros(5)
    _py = np.zeros(5)
    
    cdef double xleft, xright, ytop, ybottom
    cdef double a, S, midLeftX, midTopY, longSide
    cdef double workx[9], worky[9]

    _workx = np.zeros(9)
    _worky = np.zeros(9)

    cdef list I = [], J = [], V = []

    cdef int m, n, wr, wc
    cdef int K, k, verts, hwin

    cdef int out_M = M * N
    cdef int out_N = MM * NN

    cdef double mt, nt, ridx, cidx, side
    cdef int skip
    
    debug = False
        
    if debug:
        turtle.speed(0)
        turtle.setworldcoordinates(-N * zoom,M*2 * zoom,N*2 * zoom,-M * zoom)
        
        px[0] = -N/2
        px[1] = N/2
        px[2] = N/2
        px[3] = -N/2
        px[4] = px[0]

        py[0] = -M/2
        py[1] = -M/2
        py[2] = M/2
        py[3] = M/2
        py[4] = py[0]

        for i in range(5):
            _px[i] = px[i]
            _py[i] = py[i]

        turtle.pencolor('green')
        drawPolygon(_px, _py, 5)

        px[0] = -N/2 * zoom
        px[1] = N/2 * zoom
        px[2] = N/2 * zoom
        px[3] = -N/2 * zoom
        px[4] = px[0]

        py[0] = -M/2 * zoom
        py[1] = -M/2 * zoom
        py[2] = M/2 * zoom
        py[3] = M/2 * zoom
        py[4] = py[0]

        for i in range(5):
            _px[i] = px[i]
            _py[i] = py[i]

        turtle.pencolor('green')
        #drawPolygon(_px, _py, 5)

    # For each element in the low-resolution source
    #for m in range(M/2,M/2+1):
    #    for n in range(N/2,N/2+1):
    for m in range(M):
        for n in range(N):
            # Create pixel polygon
            #
            # The 0.5 offset is to generate a pixel around point (m, n)
            #
            side = 0.5 * zoom
             
            px[0] = - side * math.cos(yaw) - side * math.sin(yaw) + (n-N/2.)*zoom*math.cos(yaw) + N/2. 
            px[1] = + side * math.cos(yaw) - side * math.sin(yaw) + (n-N/2.)*zoom*math.cos(yaw) + N/2. 
            px[2] = + side * math.cos(yaw) + side * math.sin(yaw) + (n-N/2.)*zoom*math.cos(yaw) + N/2.
            px[3] = - side * math.cos(yaw) + side * math.sin(yaw) + (n-N/2.)*zoom*math.cos(yaw) + N/2. 
            px[4] = px[0]

            py[0] = - side * math.sin(yaw) + side * math.cos(yaw) + (n-N/2.)*zoom*math.sin(yaw) + (m-M/2.)*zoom*math.cos(yaw) + N/2. + M/2.
            py[1] = + side * math.sin(yaw) + side * math.cos(yaw) + (n-N/2.)*zoom*math.sin(yaw) + (m-M/2.)*zoom*math.cos(yaw) + N/2. + M/2.
            py[2] = + side * math.sin(yaw) - side * math.cos(yaw) + (n-N/2.)*zoom*math.sin(yaw) + (m-M/2.)*zoom*math.cos(yaw) + N/2. + M/2.
            py[3] = - side * math.sin(yaw) - side * math.cos(yaw) + (n-N/2.)*zoom*math.sin(yaw) + (m-M/2.)*zoom*math.cos(yaw) + N/2. + M/2.
            py[4] = py[0]
            
            if debug:
                for i in range(5):
                    _px[i] = px[i] - N/2.
                    _py[i] = py[i] - N/2. - M/2.

                turtle.pencolor('blue')
                drawPolygon(_px, _py, 5)

            # Find position in high-resolution frame
            nt = ((n-N/2.)*zoom*math.cos(yaw) + N/2.)
            mt = ((n-N/2.)*zoom*math.sin(yaw) + (m-M/2.)*zoom*math.cos(yaw) + N/2. + M/2.)

            # For pixels in target vicinity
            K = 0
            hwin = search_win/2
            skip = False
            for wr in range(-hwin, -hwin + search_win):
                if skip:
                   break
                for wc in range(-hwin, -hwin + search_win):
                    xleft = nt + wc - 0.
                    xright = xleft + 1.

                    ytop = mt + wr - 0.
                    ybottom = ytop + 1.

                    ridx = np.floor(mt + wr - M/2. + hwin)
                    cidx = np.floor(nt + wc + hwin)
                    
                    # Ameliorate edge effects by skipping out of the loop
                    # whenever we hit the sides
                    if ridx < 0 or ridx > (MM - 1) or \
                       cidx < 0 or cidx > (NN - 1):
                        skip = True
                        break
                    #if abs(mt - M/2 - N/2) > 20: 
                    #    skip = True
                    #    break


                    if False:
                        rx[0] = xleft
                        rx[1] = xright
                        rx[2] = xright
                        rx[3] = xleft
                        rx[4] = xleft

                        ry[0] = ytop
                        ry[1] = ytop
                        ry[2] = ybottom
                        ry[3] = ybottom
                        ry[4] = ytop

                        for i in range(5):
                            _rx[i] = rx[i] - N/2.
                            _ry[i] = ry[i] - N/2. - M/2.

                        turtle.pencolor('red')
                        drawPolygon(_rx, _ry, 5)


                    verts = poly_clip(5, px, py,
                                      xleft, xright, ytop, ybottom,
                                      workx, worky)
                    a = area(verts, workx, worky)


                    if debug and a > 0.:
                        for i in range(verts):
                            _workx[i] = workx[i] - N/2.
                            _worky[i] = worky[i] - N/2. - M/2.

                        turtle.pencolor('magenta')
                        drawPolygon(_workx, _worky, verts)

                    #print ridx, cidx, a, wr == 0, wc == 0

                    I.append(m*N + n)
                    J.append((int)(ridx * NN + cidx))
                    V.append(a)

                    K += 1

            S = 0
            for k in range(K):
                S += V[-k]

            for k in range(K):
                if S > 1e-6:
                    V[-k] /= S

    return sparse.coo_matrix((V, (I, J)), shape=(out_M, out_N)).tocsr()
   
   
