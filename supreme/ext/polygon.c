/*
   Copyright (c) 2006, Stefan van der Walt

   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

    1. Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimers.
    2. Redistributions in binary form must reproduce the above
       copyright notice in the documentation and/or other materials
       provided with the distribution.
    3. The names of the authors may not be used to endorse or promote
       products derived from this Software without specific prior
       written permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */


#ifdef __cplusplus
extern "C" {
#endif

#include "polygon.h"
#include <math.h>

#define INF HUGE_VAL
#define NEARZERO 1e-60

unsigned char pnpoly(int nr_verts, double *xp, double *yp, double x, double y)
/*
   Code from:
   http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

   Copyright (c) 1970-2003, Wm. Randolph Franklin
*/
{
    int i,j;
    unsigned char c = 0;
    for (i = 0, j = nr_verts-1; i < nr_verts; j = i++) {
        if ((((yp[i]<=y) && (y<yp[j])) ||
             ((yp[j]<=y) && (y<yp[i]))) &&
            (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))

            c = !c;
    }
    return c;
}

void npnpoly(int nr_verts, double *xp, double *yp,
             int N, double *x, double *y,
             unsigned char *result)
/*
 * For N provided points, calculate whether they are in
 * the polygon defined by vertices *xp, *yp.
 *
 * nr_verts -- number of vertices
 * *xp, *yp -- x and y coordinates of vertices
 * N        -- number of data points provided
 * *x, *y   -- data points
 */
{
    int n = 0;
    for (n = 0; n < N; n++) {
        result[n] = pnpoly(nr_verts,xp,yp,x[n],y[n]);
    }
}

void line_intersect(double x0, double y0, double x1, double y1, /* line 1 */
                    double x2, double y2, double x3, double y3, /* line 2 */
                    struct POI *p)
/*
  Calculate the point of intersection between two lines.

  See Paul Bourke's astronomy page at
  http://astronomy.swin.edu.au/~pbourke/geometry
*/
{
    double d, ua, ub;

    d = (y3 - y2)*(x1 - x0) - (x3 - x2)*(y1 - y0);
    ua = (x3 - x2)*(y0 - y2) - (y3 - y2)*(x0 - x2);
    ub = (x1 - x0)*(y0 - y2) - (y1 - y0)*(x0 - x2);

    if (d == 0) {
        p->x = 0;
        p->y = 0;

        if ((ua == 0) && (ub == 0))
            p->type = 3;
        else
            p->type = 2;
        return;
    }

    ua = ua/d;
    ub = ub/d;

    p->x = x0 + ua*(x1 - x0);
    p->y = y0 + ua*(y1 - y0);
    if ((ua >= 0) && (ua <= 1) && (ub >= 0) && (ub <= 1))
        p->type = 0;
    else
        p->type = 1;

    return;
}

int poly_clip(int N, double* x, double* y,
              double xleft, double xright, double ytop, double ybottom,
              double* workx, double* worky)
/* Clip a closed polygon of N vertices (xp,yp) to the specified
   bounding box using the Liang-Barsky algorithm. The resulting
   polygon of M vertices are placed in 'work_x' and 'work_y' (which
   must be of length 2*N-1) and M is returned.

   See You-Dong Lian and Brian A. Barsky,
       An Analysis and Algorithm for Polygon Clipping,
       Communications of the ACM, Vol 26, No 11, November 1983

   The algorithm is a translation of the Pascal code found in the
   article, and was modified to includes fixes from

   Anti-Grain Geometry - Version 2.4
   Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)

*/
{
    double deltax, deltay, xin, yin, xout, yout, tinx, tiny;
    double toutx, touty, tin1, tin2, tout1;
    int i, M;

    M = 0;
    for (i = 0; i < N-1; i++) { /* edge V[i]V[i+1] */
        deltax = x[i+1] - x[i];
        deltay = y[i+1] - y[i];

        /* bump off the vertical */
        if (deltax == 0) {
            deltax = x[i] > xleft ?-NEARZERO : NEARZERO;
        }

        /* bump off the horizontal */
        if (deltay == 0) {
            deltay = y[i] > ytop ? -NEARZERO : NEARZERO;
        }

        if (deltax > 0) { /* l[i] points to the right */
            xin = xleft;
            xout = xright;
        }
        else { /* l[i] points to the left */
            xin = xright;
            xout = xleft;
        }

        if (deltay > 0) { /* l[i] points up */
            yin = ybottom;
            yout = ytop;
        } else { /* l[i] points down */
            yin = ytop;
            yout = ybottom;
        }

        /* start fix from AGG */
        tinx = (xin - x[i])/deltax;
        tiny = (yin - y[i])/deltay;
        /* end fix */

        if (tinx < tiny) { /* first entry at x then y */
            tin1 = tinx;
            tin2 = tiny;
        } else { /* first entry at y then x */
            tin1 = tiny;
            tin2 = tinx;
        }

        if (tin1 <= 1) { /* case 2 or 3 or 4 or 6 */
            if (0 < tin1) { /* case 5 -- turning vertex */
                workx[M] = xin;
                worky[M] = yin;
                M++;
            }

            if (tin2 <= 1) { /* case 3 or 4 or 6 */
                toutx = (xout - x[i])/deltax;
                touty = (yout - y[i])/deltay;

                tout1 = (toutx < touty) ? toutx : touty;

                if ((tin2 > 0) || (tout1 > 0)) { /* case 4 or 6 */
                    if (tin2 <= tout1) { /* case 4 -- visible segment */
                        if (tin2 > 0) { /* V[i] outside window */
                            if (tinx > tiny) { /* vertical boundary */
                                workx[M] = xin;
                                worky[M] = y[i] + tinx*deltay;
                                M++;
                            } else { /* horisontal boundary */
                                workx[M] = x[i] + tiny*deltax;
                                worky[M] = yin;
                                M++;
                            }
                        }

                        if (tout1 < 1) {/* V[i+1] outside window */
                            if (toutx < touty) { /* vertical boundary */
                                workx[M] = xout;
                                worky[M] = y[i] + toutx*deltay;
                                M++;
                            } else { /* horisontal boundary */
                                workx[M] = x[i] + touty*deltax;
                                worky[M] = yout;
                                M++;
                            }
                        } else { /* V[i+1] inside window */
                            workx[M] = x[i+1];
                            worky[M] = y[i+1];
                            M++;
                        }
                    } else { /* case 6 -- turning vertex */
                        if (tinx > tiny) { /* second entry at x */
                            workx[M] = xin;
                            worky[M] = yout;
                            M++;
                        } else { /* second entry at y */
                            workx[M] = xout;
                            worky[M] = yin;
                            M++;
                        }
                    }
                } /* case 4 or 6 */
            } /* case 3, 4 or 6 */
        } /* case 2, 3, 4 or 6 */
    } /* edge V[i]V[i+1] */

    /* close polygon */
    workx[M] = workx[0];
    worky[M] = worky[0];
    M++;

    return M;
}

int yaw_poly_clip(int N, double* x, double* y,
              double xleft, double xright, double ytop, double ybottom,
              double* workx, double* worky)
/* Clip a closed polygon of N vertices (xp,yp) to the specified
   bounding box using the Liang-Barsky algorithm. The resulting
   polygon of M vertices are placed in 'work_x' and 'work_y' (which
   must be of length 2*N-1) and M is returned.

   See You-Dong Lian and Brian A. Barsky,
       An Analysis and Algorithm for Polygon Clipping,
       Communications of the ACM, Vol 26, No 11, November 1983

   The algorithm is a translation of the Pascal code found in the
   article, and was modified to includes fixes from

   Anti-Grain Geometry - Version 2.4
   Copyright (C) 2002-2005 Maxim Shemanarev (McSeem)

*/
{
    double deltax, deltay, xin, yin, xout, yout, tinx, tiny;
    double toutx, touty, tin1, tin2, tout1;
    int i, M;

    M = 0;
    for (i = 0; i < N-1; i++) { /* edge V[i]V[i+1] */
        deltax = x[i+1] - x[i];
        deltay = y[i+1] - y[i];

        /* bump off the vertical */
        if (deltax == 0) {
            deltax = x[i] > xleft ?-NEARZERO : NEARZERO;
        }

        /* bump off the horizontal */
        if (deltay == 0) {
            deltay = y[i] > ytop ? -NEARZERO : NEARZERO;
        }

        if (deltax > 0) { /* l[i] points to the right */
            xin = xleft;
            xout = xright;
        }
        else { /* l[i] points to the left */
            xin = xright;
            xout = xleft;
        }

        if (deltay < 0) { /* l[i] points up */
            yin = ybottom;
            yout = ytop;
        } else { /* l[i] points down */
            yin = ytop;
            yout = ybottom;
        }

        /* start fix from AGG */
        tinx = (xin - x[i])/deltax;
        tiny = (yin - y[i])/deltay;
        /* end fix */

        if (tinx < tiny) { /* first entry at x then y */
            tin1 = tinx;
            tin2 = tiny;
        } else { /* first entry at y then x */
            tin1 = tiny;
            tin2 = tinx;
        }

        if (tin1 <= 1) { /* case 2 or 3 or 4 or 6 */
            if (0 < tin1) { /* case 5 -- turning vertex */
                workx[M] = xin;
                worky[M] = yin;
                M++;
            }

            if (tin2 <= 1) { /* case 3 or 4 or 6 */
                toutx = (xout - x[i])/deltax;
                touty = (yout - y[i])/deltay;

                tout1 = (toutx < touty) ? toutx : touty;

                if ((tin2 > 0) || (tout1 > 0)) { /* case 4 or 6 */
                    if (tin2 <= tout1) { /* case 4 -- visible segment */
                        if (tin2 > 0) { /* V[i] outside window */
                            if (tinx > tiny) { /* vertical boundary */
                                workx[M] = xin;
                                worky[M] = y[i] + tinx*deltay;
                                M++;
                            } else { /* horisontal boundary */
                                workx[M] = x[i] + tiny*deltax;
                                worky[M] = yin;
                                M++;
                            }
                        }

                        if (tout1 < 1) {/* V[i+1] outside window */
                            if (toutx < touty) { /* vertical boundary */
                                workx[M] = xout;
                                worky[M] = y[i] + toutx*deltay;
                                M++;
                            } else { /* horisontal boundary */
                                workx[M] = x[i] + touty*deltax;
                                worky[M] = yout;
                                M++;
                            }
                        } else { /* V[i+1] inside window */
                            workx[M] = x[i+1];
                            worky[M] = y[i+1];
                            M++;
                        }
                    } else { /* case 6 -- turning vertex */
                        if (tinx > tiny) { /* second entry at x */
                            workx[M] = xin;
                            worky[M] = yout;
                            M++;
                        } else { /* second entry at y */
                            workx[M] = xout;
                            worky[M] = yin;
                            M++;
                        }
                    }
                } /* case 4 or 6 */
            } /* case 3, 4 or 6 */
        } /* case 2, 3, 4 or 6 */
    } /* edge V[i]V[i+1] */

    /* close polygon */
    workx[M] = workx[0];
    worky[M] = worky[0];
    M++;

    return M;
}


void tf_polygon(int N, double* xp, double* yp, double* tf_M) {
    int i;
    double x_new, y_new, z;
    for (i = 0; i < N; i++) {
        z = tf_M[6] * xp[i] + tf_M[7] * yp[i] + tf_M[8];
        x_new = (tf_M[0] * xp[i] + tf_M[1] * yp[i] + tf_M[2]) / z;
        y_new = (tf_M[3] * xp[i] + tf_M[4] * yp[i] + tf_M[5]) / z;
        xp[i] = x_new;
        yp[i] = y_new;
    }
}

double area(int N, double* px, double* py) 
{
    double A = 0;
    int n;
    for (n = 0; n < N-1; n++)
        A += px[n] * py[n+1] - px[n+1] * py[n];
    if (A < 0) A *= -1;
    return 0.5 * A;
}

double area_gauss(int N, double* px, double* py, double gx, double gy, double sigma) 
{   
    double A = 0;
    double minx = HUGE;
    double miny = HUGE;
    double maxx = -HUGE;
    double maxy = -HUGE;
    double x, y;
    double wx = 0.1;
    double wy = 0.1;    
    int i;

    for (i = 0; i < N; i++)
    {
      if (px[i] > maxx) 
        maxx = px[i];
      if (py[i] > maxy)
        maxy = py[i];
      if (px[i] < minx)
        minx = px[i];
      if (py[i] < miny)
        miny = py[i];
    }

    for (y = miny + wy/2; y < maxy; y += wy )
        for (x = minx + wx/2; x < maxx; x += wx )
        {
            //printf("x=%f, y=%f\n", x-gx,y-gy);
            if (pnpoly(N, px, py, x, y))
                A += //exp(-(x - gx)*(x - gx)/(sigma*sigma)) *
                     exp(-(y - gy)*(y - gy)/(sigma*sigma)) *
                     wx * wy;   
        }
    //printf("%f\n", A);
    return A;
}

void interp_transf_polygon(int target_rows, int target_cols, unsigned char* target,
                           int out_rows, int out_cols, double* out,
                           double* inv_tf_M)
{
    double rx[5], ry[5];
    double xleft,xright,ytop,ybottom;
    double wweight[9],wsum,val,intensity[9];
    double workx[9],worky[9];
    int oc,or,wr,wc;
    int i;
    int verts;
    /* For each element in the high-resolution output */
    for (oc = 0; oc < out_cols; oc++)
        for (or = 0; or < out_rows; or++) {
            /* Create pixel polygon */
            rx[0] = oc; rx[1] = oc+1; rx[2] = oc+1;
            rx[3] = oc; rx[4] = oc;

            ry[0] = or; ry[1] = or; ry[2] = or+1;
            ry[3] = or+1; ry[4] = or;

            /* Calculate coordinates in target frame */
            tf_polygon(5,rx,ry,inv_tf_M);

            /*            _print_poly(5,rx,ry);*/

            /* For surrounding 9 pixels */
            for (wr = 0; wr < 3; wr++)
                for (wc = 0; wc < 3; wc++) {
                    xleft = floor(rx[0]) + wc - 1;
                    xright = xleft+1;
                    ybottom = floor(ry[0]) + wr - 1;
                    ytop = ybottom + 1;

                    if ((xleft >= 0) && (xleft < target_cols) &&
                        (ybottom >= 0) && (ybottom < target_rows)) {
                        verts = poly_clip(5,rx,ry,
                                          xleft,xright,ytop,ybottom,
                                          workx,worky);
                        wweight[wr*3 + wc] = area(verts,workx,worky);
                        intensity[wr*3 + wc] =
                            target[((int)ybottom)*target_cols + (int)xleft];
                    } else {
                        wweight[wr*3 + wc] = 0;
                        intensity[wr*3 + wc] = 0;
                    }
                }

            val = 0;
            wsum = 0;
            for (i = 0; i < 9; i++) {
                val += wweight[i]*intensity[i];
                wsum += wweight[i];
            }
            if (wsum > 1e-15)
                out[or*out_cols + oc] = val / wsum;
            else
                out[or*out_cols + oc] = 0;
        }
}

#ifdef __cplusplus
}
#endif
