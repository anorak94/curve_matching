import numpy as np
import matplotlib.pyplot as plt


def get_psi(ds1, ds2):
    """
    given the two ds segments computes the cosine and sine of the angle between the matching 
    """
    hyp = (ds1**2 + ds2**2)**0.5
    cos_psi = ds1 / hyp
    sin_psi = ds2 / hyp
    
    return cos_psi, sin_psi
    
def get_delta_s(p1, p2):
    """
    computes the ds between two points on a curve 
    """
    x1, y1 = p1
    x2, y2 = p2
    ds = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return ds

def get_kappa_and_derivatives(a):
    """
    returns the curvature and gradients of a curve segment 
    """
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature, [dx_dt, dy_dt, d2x_dt2, d2y_dt2] 


def get_segments_from_pt_list(p1):
    seg_list = []
    curv_list = []
    for i in range(1, len(p1)):
        strt = p1[i-1]
        end  = p1[i]
        ds   = get_delta_s(strt, end)
        seg_list.append(ds)
        
    return seg_list

def get_unique_elements(p1):
    """
    returns only unique pts from the pt list 
    """
    pint = []
    pset = set()
    for pi in p1:
        if (pi[0], pi[1]) in pset:
            pass
        else:
            pint.append(pi)
            pset.add((pi[0], pi[1]))
    return np.asarray(pint)
    
def get_tup_list(i, j):
    """
    get valid tup list from all possible tuples 
    """
    str_tup_list = [(i, i), (i-1, i), (i-1, i), (i-2, i), (i-3, i), (i-3, i), (i-2, i), (i-1, i), (i-1, i)]
    end_tup_list = [(j-1, j), (j, j), (j-1, j), (j-1, j), (j-1, j), (j-2, j), (j-3, j), (j-3, j), (j-2, j)]
    s1_list = []
    s2_list = []
    for (str_tup, end_tup) in zip(str_tup_list, end_tup_list):
        s1, s2 = str_tup
        u1, u2 = end_tup
        if s1<0 or s2<0 or u1<0 or u2<0:
            pass
        else:
            s1_list.append(str_tup)
            s2_list.append(end_tup)
    
    return s1_list, s2_list
        

def plot_cost_map(i, j, p1_int, p2_int):
    
    """
    plots the possible cost maps of the decision to be taken from i, j 
    """

    s1_list, s2_list = get_tup_list(i, j)

    for (str_tup, end_tup) in zip(s1_list, s2_list):
        i1, i2 = str_tup  ## here we match the segments i1-i2 in the first curve with j1-j2 in the second curve 
        j1, j2 = end_tup 
        #
        print (f"{i1} to  {i2} in curve 1")
        print (f"{j1} to  {j2} in curve 2 ")
        p1_pts = p1_int[i1:i2+1]  ## these will be the corresponding pts in both the curve 
        p2_pts = p2_int[j1:j2+1]

        f, a  = plt.subplots(1, 2, figsize = (12, 8))
        a[0].scatter(p1_int[:, 0], p1_int[:, 1], c = "b")   ## i is blue 
        a[1].scatter(p2_int[:, 0], p2_int[:, 1], c = "y")   ## j is yellow 
        a[0].plot(p1_int[:, 0], p1_int[:, 1])
        a[0].set_axis_off()
        a[1].set_axis_off()
        a[1].plot(p2_int[:, 0], p2_int[:, 1])
        #print (p1_pts)
        if len(p1_pts) == 1:
            a[0].scatter(p1_pts[:, 0], p1_pts[:, 1], c = "r")
        else:
            a[0].plot(p1_pts[:, 0], p1_pts[:, 1], c = "r")
        if len(p2_pts) == 1:
            a[1].scatter(p2_pts[:, 0], p2_pts[:, 1], c = "r")
        else: 
            a[1].plot(p2_pts[:, 0], p2_pts[:, 1], c = "r")



        plt.show()
        plt.close("all")
        
def get_tangent_vector(a):
    """
    returns the tangent vectors of a curve segment 
    """
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    
    
    return np.asarray([dx_dt, dy_dt]).T

def matching_cost(i1, i2, j1, j2, p1_int, p2_int, kappa1, kappa2, R):

    """
    computes the cost of matching the given curve segments 
    """
    
    pt1_list = p1_int[i1:i2+1]  ## these will be the corresponding pts in both the curve 
    pt2_list = p2_int[j1:j2+1]
    
    ## pt1_list, pt2_list
    if len(pt1_list) == 1:
        ## do something 
        ds2 =  get_delta_s(pt2_list[0], pt2_list[-1])
        cost = ds2*(1 +  R*abs(kappa2[j2]+kappa2[j2-1])*0.5)

    elif len(pt2_list) == 1:
        ## do something 
        ds1 =  get_delta_s(pt1_list[0], pt1_list[-1])
        cost = ds1*(1 +  R*abs(kappa1[i2-1]+kappa1[i2])*0.5)

    elif len(pt1_list) > 1 and len(pt2_list) > 1 :

        ds_dt1 = get_tangent_vector(pt1_list) ## compute
        ds_dt2 = get_tangent_vector(pt2_list) ## compute 

        Ta = ds_dt1[0]
        Tb = ds_dt1[-1]

        Tb_ = ds_dt2[-1]

        normTa = np.linalg.norm(Ta)
        normTb = np.linalg.norm(Tb)
        normTb_ = np.linalg.norm(Tb_)
        
        if normTb == 0.0 or normTa == 0.0 or normTb_ == 0:
            print (pt1_list, pt2_list)
        
        cos_theta1 = np.dot(Ta, Tb)/(normTa*normTb)
        cos_theta2 = np.dot(Ta, Tb_)/(normTa*normTb_)
        
        
        dtheta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
        dtheta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))
        
        
        
        ds1 = get_delta_s(pt1_list[0], pt1_list[-1])
        ds2 = get_delta_s(pt2_list[0], pt2_list[-1])

        cost = abs(ds1-ds2) + R*abs(dtheta1 - dtheta2)
        #print (pt1_list, pt2_list)
    
    return cost 


def plot_points(p1, p2):
    f, a = plt.subplots(1, 3, figsize = (12, 4))
    a[0].invert_yaxis()
    a[0].scatter(p1[:, 0], p1[:, 1], c = "r")
    a[0].set_axis_off()
    a[1].invert_yaxis()
    a[1].scatter(p2[:, 0], p2[:, 1], c = "y")
    a[1].set_axis_off()
    a[2].invert_yaxis()
    a[2].scatter(p2[:, 0], p2[:, 1], c = "y")
    a[2].scatter(p1[:, 0], p1[:, 1], c = "r")
    a[2].set_axis_off()
    plt.show()


def get_matches(dist, predecessor):

    src1, src2 = dist.shape[0]-1, dist.shape[1]-1
    #c = 0
    global_path = [(src1, src2)]
    while((src1, src2) != (0, 0)):
        global_path.append(predecessor[src1][src2])
        src1, src2 = predecessor[src1][src2]
        #c = c+1
        #if c > 2500:
        #    break
    return global_path


def plot_matches(p1_int, p2_int, global_path, offset, nstart, npts):
    p1_x , p1_y = p1_int[:, 0] + offset, p1_int[:, 1]
    f , a = plt.subplots(1, 1, figsize = (16 , 8)) 
    a.invert_yaxis()
    a.scatter(p2_int[:, 0], p2_int[:, 1], c = "y")
    a.scatter(p1_x, p1_y, c = "r")
    a.set_axis_off()

    arr_p = {"width": 1e-6, "head_width":1e-4}
    for (m1, m2) in global_path[nstart:nstart+npts]:
        a1x, a1y = p1_x[m1-1], p1_y[m1-1]
        dx, dy = p2_int[m2-1][0] - a1x, p2_int[m2-1][1] - a1y
        a.arrow(a1x, a1y, dx, dy, alpha = 0.25, fc = "red", **arr_p )
    plt.show()
