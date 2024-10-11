import numpy as np
import logging
import torch
from tqdm import tqdm

def derivatives(phases, delta_x, delta_y):
    dims = phases.shape
    #print(dims)
    if len(dims) == 2:
        logging.error("derivatives.py | derivatives() needs shape (b,w,h)")
        exit()
    else:
        phases = torch.reshape(phases, (dims[0], dims[-1] * dims[-2]))
    logging.debug("derivatives() | phase.shape = {}".format(phases.shape))

    delta_x = delta_x.to(phases.device)
    delta_y = delta_y.to(phases.device)

    mat1 = torch.flatten(torch.tensor([ [0,0,0],
                                        [-1,0,1],
                                        [0,0,0] ]).float()).to(phases.device)

    mat2 = torch.flatten(torch.tensor([ [0,1,0],
                                        [0,0,0],
                                        [0,-1,0] ]).float()).to(phases.device)

    mat3 = torch.flatten(torch.tensor([ [0,0,0],
                                        [1,-2,1],
                                        [0,0,0] ]).float()).to(phases.device)

    mat4 = torch.flatten(torch.tensor([ [0,1,0],
                                        [0,-2,0],
                                        [0,1,0] ]).float()).to(phases.device)

    mat5 = torch.flatten(torch.tensor([ [1,0,-1],
                                        [0,0,0],
                                        [-1,0,1] ]).float()).to(phases.device)

    mat1 = torch.reshape(mat1, (1,9))
    mat2 = torch.reshape(mat2, (1,9))
    mat3 = torch.reshape(mat3, (1,9))
    mat4 = torch.reshape(mat4, (1,9))
    mat5 = torch.reshape(mat5, (1,9))

    # The following math is just batch wise dot product. This is done to keep us
    # from needing to do weird indexing to get the derivatives. The indexing used here
    # is adapted from https://stackoverflow.com/questions/69230570/how-to-do-batched-dot-product-in-pytorch
    fx = ((phases / (2*delta_x)) * mat1[None, ...]).sum(dim=-1).squeeze()
    fy = ((phases / (2*delta_y)) * mat2[None, ...]).sum(dim=-1).squeeze()
    
    fxx = ((phases/(delta_x**2)) * mat3[None, ...]).sum(dim=-1).squeeze()
    fyy = ((phases/(delta_y**2)) * mat4[None, ...]).sum(dim=-1).squeeze()

    fxy = ((phases/(4 * delta_x**2)) * mat5[None, ...]).sum(dim=-1).squeeze()

    #return {"fx":fx, "fy":fy, "fxx":fxx, "fyy":fyy, "fxy":fxy}
    return fx, fy, fxx, fyy, fxy

def rot_mat_linear(derivatives):
    fx = derivatives['fx']
    fy = derivatives['fy']
    logging.debug("rot_mat_linear() | fx,fy = {},{}".format(fx,fy))

    if (fx**2 + fy**2) != 0:
        temp = fx**2 + fy**2
        temp_2 = torch.sqrt(temp + 1e-9)
        scale = torch.div(1,temp_2)
        logging.debug("rot_mat_linear() | scale = {}".format(scale))

        #This is here just so we can see the shape of what we return
        #rot_mat_flat = scale * [fy,fx,-fx,fy]
        fy_scale = scale*fy
        fx_scale = scale*fx
        fx_neg = (-1)*scale*fx
        
        #return scale*fy, scale*fx, (-1)*scale*fx, scale*fy
        return fy_scale, fx_scale, fx_neg, fy_scale
    else: 
        logging.warning("rot_mat_linear() | Encountered edge case!!!")
        p1 = fy
        p2 = 1**fx
        p3 = -(1)**fx
        p4 = fy
        #return fy, 1**fx, -1**fx, fy
        return p1, p2, p3, p4
    
def mag_grad(derivatives):
    fx = derivatives['fx']
    fy = derivatives['fy']
    logging.debug("mag_grad() | fx,fy = {},{}".format(fx,fy))
    temp = fx**2 + fy**2
    #magnitude = torch.sqrt(fx**2 + fy**2)
    magnitude = torch.sqrt(temp + 1e-9)
    logging.debug("mag_grad() | magnitude : {}".format(magnitude))
    return magnitude

def rot_mat_quad(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    if fxx+fyy+fxy != 0:
        temp = (fxx-fyy)**2 + 4*(fxy**2)
        #vp0 = fxx - fyy - torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))
        vp0 = fxx - fyy - torch.sqrt(temp + 1e-9)
        vp1 = 2*fxy
        
        temp2 = vp0**2 + vp1**2
        denominator1 = torch.sqrt(temp + 1e-9) + 1e-9
        #denominator1 = torch.sqrt(vp0**2 + vp1**2) + 1e-9
        vp0_adj = vp0 / denominator1
        vp1_adj = vp1 / denominator1
        
        temp = (fxx-fyy)**2 + 4*(fxy**2)        
        #vq0 = fxx - fyy + torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))
        vq0 = fxx - fyy + torch.sqrt(temp + 1e-9)
        vq1 = 2*fxy
        #print(f"vq0 = {vq0}, {vq0.dtype}")
        #print(f"vq1 = {vq1}, {vq0.dtype}")

        temp = vq0**2 + vq1**2
        denominator2 = torch.sqrt(temp + 1e-9) + 1e-9

        vq0_adj = vq0 / denominator2
        vq1_adj = vq1 / denominator2

        rquad = torch.stack((vp0_adj, vp1_adj, vq0_adj, vq1_adj))
        rquad = rquad.view(2,2)
        rquad = rquad.transpose(0,1)
    else:
        logging.warning("rot_mat_quad() | Encountered edge case!!!")
        rquad = torch.stack((fxx, -1**fyy, 1**fxy, fxx))
        rquad = rquad.view(2,2)
        rquad = rquad.transpose(0,1)
    return rquad

def fpp(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    temp = (fxx - fyy)**2 + 4*(fxy**2)
    fpp = 0.5*(fxx + fyy - torch.sqrt(temp + 1e-9)) 
    #fpp = 0.5*(fxx + fyy - torch.sqrt((fxx-fyy)**2 + 4*(fxy**2))) 
    return fpp

def fqq(derivatives):
    fxx = derivatives['fxx']
    fyy = derivatives['fyy']
    fxy = derivatives['fxy']
    temp = (fxx-fyy)**2 + 4*(fxy**2)
    fqq = 0.5*(fxx + fyy + torch.sqrt(temp + 1e-9))
    #fqq = 0.5*(fxx + fyy + torch.sqrt((fxx-fyy)**2 + 4*(fxy**2)))
    return fqq

def rot_angle(rot_matrix):
    rot_matrix = rot_matrix
    angle = torch.arctan2(rot_matrix[1], rot_matrix[0])
    return angle

def average_phase(phases):

    return torch.mean(phases.view(-1, 9), dim = 1)

# Gather curvature info from phase
 
def get_curv(phase):
    #print(f"get_curv: phases.shape = {phases.shape}") 
    der = derivatives(phase, torch.tensor(1),torch.tensor(1))
    curvature = {}
    curvature['magnitude'] = mag_grad(der)
    curvature['linear_rotation'] = rot_angle(rot_mat_linear(der))
    curvature['fpp'] = fpp(der)
    curvature['fqq'] = fqq(der)
    curvature['quad_rotation'] = rot_angle(rot_mat_quad(der).flatten())
    curvature['avgs'] = torch.unsqueeze(average_phase(phase), dim = 1)
    return curvature

def get_curv_train(phases):    
    #print(f"get_curv_train: phases.shape = {phases.shape}")
    
    magnitude = []
    linear_rotation = []
    fpp_ = []
    fqq_ = []
    quad_rotation = []
    average = []

    for p in phases:
        der = derivatives(p.unsqueeze(dim=0), torch.tensor(1), torch.tensor(1))
        magnitude.append(mag_grad(der))
        linear_rotation.append(rot_angle(rot_mat_linear(der)))
        fpp_.append(fpp(der))
        fqq_.append(fqq(der))
        quad_rotation.append(rot_angle(rot_mat_quad(der).flatten()))
        average.append(average_phase(p))

    mags = torch.stack(magnitude).unsqueeze(dim=1)
    rot_lin = torch.stack(linear_rotation).unsqueeze(dim=1)
    fpp_val = torch.stack(fpp_).unsqueeze(dim=1)
    fqq_val = torch.stack(fqq_).unsqueeze(dim=1)
    rot_quad = torch.stack(quad_rotation).unsqueeze(dim=1)
    avgs = torch.stack(average)

    return torch.hstack([mags, rot_lin, fpp_val, fqq_val, rot_quad, avgs])   

def get_der_train(phases):
    der = []
    average = []
    for p in phases:
        der.append(torch.stack(derivatives(p.unsqueeze(dim=0), torch.tensor(1), torch.tensor(1))).unsqueeze(dim=0))
        average.append(average_phase(p).unsqueeze(dim=0))

    der = torch.cat(der)
    average = torch.cat(average)
    
    combined = torch.cat((der,average), dim=-1)

    return combined


###########################################################################################
###########################################################################################
###########################################################################################
def test_derivatives(values):
    kov = kov_derivatives(values)
    values = torch.reshape(values, (1,3,3))
    test = derivatives(values, torch.tensor(1), torch.tensor(1))
    
    test_list = []
    for k in test:
        test_list.append(test[k].detach().numpy())

    test_list = np.asarray(test_list)
    logging.debug("Dr. K derivatives : {}".format(kov))
    logging.debug("Our derivatives : {}".format(test_list))
    logging.debug("Do all match?? : {}".format((test_list == kov).all()))
    logging.debug("")
    

def test_rlin(values):
    kov_der = kov_derivatives(values)
    kov_rlin = kov_rot_mat_linear(kov_der).flatten()
    kov_rlin = torch.from_numpy(kov_rlin)
    kov_rlin = torch.split(kov_rlin, 1)

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, torch.tensor(1),torch.tensor(1))
    rlin = rot_mat_linear(derivs)

    logging.debug("Dr. K Rlin : {}".format((kov_rlin)))
    logging.debug("Our Rlin : {}".format(rlin))
    logging.debug("Do all match?? : {}".format((rlin == kov_rlin)))
    logging.debug("")


def test_mag_grad(values):
    kov_der = kov_derivatives(values)
    kov_mag = kov_mag_grad(kov_der)
    kov_mag = torch.tensor(kov_mag)

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, 1,1)
    mag = mag_grad(derivs)

    logging.debug("Dr. K mag_grad : {}".format(kov_mag))
    logging.debug("Our mag_grad : {}".format(mag))
    logging.debug("Do all match?? : {}".format((mag == kov_mag)))
    logging.debug("")
    

def test_rquad(values):
    kov_der = kov_derivatives(values)
    kov_rquad = kov_rot_mat_quad(kov_der)
    kov_rquad = torch.from_numpy(kov_rquad)

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, torch.tensor(1),torch.tensor(1))
    rquad = rot_mat_quad(derivs)
    
    logging.debug("Dr. K rot_quad : {}".format(kov_rquad))
    logging.debug("Our mag_grad : {}".format(rquad))
    logging.debug("Do all match?? : {}".format((kov_rquad == rquad).all()))
    logging.debug("")


def test_fpp(values):
    kov_der = kov_derivatives(values)
    kov_fpp_val = kov_fpp(kov_der)
    kov_fpp_val = torch.tensor(kov_fpp_val)

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, torch.tensor(1),torch.tensor(1))
    our_fpp = fpp(derivs)
    
    logging.debug("Dr. K fpp : {}".format(kov_fpp_val))
    logging.debug("Our mag_grad : {}".format(our_fpp))
    logging.debug("Do all match?? : {}".format((kov_fpp_val == our_fpp)))
    logging.debug("")

def test_fqq(values):
    kov_der = kov_derivatives(values)
    kov_fqq_val = kov_fqq(kov_der)
    kov_fqq_val = torch.tensor(kov_fqq_val)

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, torch.tensor(1),torch.tensor(1))
    our_fqq = fqq(derivs)
    
    logging.debug("Dr. K fpp : {}".format(kov_fqq_val))
    logging.debug("Our mag_grad : {}".format(our_fqq))
    logging.debug("Do all match?? : {}".format((kov_fqq_val == our_fqq)))
    logging.debug("")

def test_rot_angle(values):
    kov_der = kov_derivatives(values)
    kov_rlin = kov_rot_mat_linear(kov_der)
    kov_rquad = kov_rot_mat_quad(kov_der)

    kov_angle_lin = kov_rot_angle(kov_rlin)
    kov_angle_quad = kov_rot_angle(kov_rquad)

    logging.debug("Dr. K linear angle : {}".format(kov_angle_lin))
    logging.debug("Dr. K quad angle : {}".format(kov_angle_quad))

    values = torch.reshape(values, (1,3,3))
    derivs = derivatives(values, 1,1)
    rlin = rot_mat_linear(derivs)
    rquad = rot_mat_quad(derivs)
    angle_lin = rot_angle(rlin)
    angle_quad = rot_angle(rquad.flatten())

    logging.debug("Our linear angle : {}".format(angle_lin))
    logging.debug("Our quad angle : {}".format(angle_quad))

    logging.debug("Linear match?? {}".format(kov_angle_lin == angle_lin))
    logging.debug("Quad angle match?? {}".format(kov_angle_quad == angle_quad))

def test_get_curv(values):
    kov_curv = kov_get_curv(values)
    
    logging.debug("Dr. K curvature dictionary: ")
    log_dict(kov_curv)
    values = torch.reshape(values, (1,3,3))

    curv = get_curv(values)
    logging.debug("Our curvature dictionary: ")
    log_dict(curv)
    logging.debug("Dictionaries match?? {}".format(kov_curv == curv))

def log_dict(dictionary):
    for k in dictionary:
        logging.debug("     {} : {}".format(k, dictionary[k]))


def test_batch_support(values):
    batch = [] 
    for v in values:
        v = v.unsqueeze(dim=0)
        batch.append(v)
    batch = torch.cat(batch, dim=0)

    derivs = derivatives(batch, 1,1)


def test_for_nans_infs():
    phases = torch.rand(10000,3,3, requires_grad=True)    
    for p in tqdm(phases):
        curv = get_curv(p.view(1,3,3))
        check_for_nans_infs(curv)
    

def check_for_nans_infs(dictionary):
    for k in dictionary:
        if torch.isnan(dictionary[k]):
            print("FOUND A NAN - starting embed")
            from IPython import embed; embed()
        if torch.isinf(dictionary[k]):
            print("FOUND AN INF - starting embed")
            from IPython import embed; embed()

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    test_lin_slope_x = torch.tensor( [  [-1, 0, 1],
                                        [-1, 0, 1], 
                                        [-1, 0, 1]], dtype = torch.float32, requires_grad=True )

    test_lin_slope_y = torch.tensor( [  [-1,-1,-1], 
                                        [0, 0, 0], 
                                        [1, 1, 1]], dtype = torch.float32, requires_grad=True )

    test_curve_x = torch.tensor( [  [1, 0, 1], 
                                    [1, 0, 1], 
                                    [1, 0, 1]], dtype = torch.float32, requires_grad=True )
    
    test_curve_y = torch.tensor( [  [-1,-1,-1], 
                                    [0, 0, 0], 
                                    [-1,-1,-1]], dtype = torch.float32, requires_grad=True )
    

    test_saddle = torch.tensor( [   [1, 0, -1], 
                                    [0, 0, 0], 
                                    [-1, 0, 1]], dtype = torch.float32, requires_grad=True )


    print("###################")
    print("Testing Derivatives")
    print("###################")
    test_derivatives(test_lin_slope_x)
    test_derivatives(test_lin_slope_y)
    test_derivatives(test_curve_x)
    test_derivatives(test_curve_y)
    test_derivatives(test_saddle)
    #
    #print("###################")
    #print("Testing Rlinear")
    #print("###################")
    #test_rlin(test_lin_slope_x)
    #test_rlin(test_lin_slope_y)
    #test_rlin(test_curve_x)
    #test_rlin(test_curve_y)
    #test_rlin(test_saddle)

    #print("###################")
    #print("Testing Magnitude")
    #print("###################")
    #test_mag_grad(test_lin_slope_x)
    #test_mag_grad(test_lin_slope_y)
    #test_mag_grad(test_curve_x)
    #test_mag_grad(test_curve_y)
    #test_mag_grad(test_saddle)

    #print("###################")
    #print("Testing Rquad")
    #print("###################")
    #test_rquad(test_lin_slope_x)
    #test_rquad(test_lin_slope_y)
    #test_rquad(test_curve_x)
    #test_rquad(test_curve_y)
    #test_rquad(test_saddle)

    #print("###################")
    #print("Testing fpp")
    #print("###################")
    #test_fpp(test_lin_slope_x)
    #test_fpp(test_lin_slope_y)
    #test_fpp(test_curve_x)
    #test_fpp(test_curve_y)
    #test_fpp(test_saddle)

    #print("###################")
    #print("Testing fqq")
    #print("###################")
    #test_fqq(test_lin_slope_x)
    #test_fqq(test_lin_slope_y)
    #test_fqq(test_curve_x)
    #test_fqq(test_curve_y)
    #test_fqq(test_saddle)


    #print("###################")
    #print("Testing rot_angle")
    #print("###################")
    #test_rot_angle(test_lin_slope_x)
    #test_rot_angle(test_lin_slope_y)
    #test_rot_angle(test_curve_x)
    #test_rot_angle(test_curve_y)
    #test_rot_angle(test_saddle)


    #print("###################")
    #print("Testing get_curv")
    #print("###################")
    #test_get_curv(test_lin_slope_x)
    #test_get_curv(test_lin_slope_y)
    #test_get_curv(test_curve_x)
    #test_get_curv(test_curve_y)
    #test_get_curv(test_saddle)


    #print("###################")
    #print("Testing batch support")
    #print("###################")
    
    #test_batch_support([test_lin_slope_x, 
    #                    test_lin_slope_y,
    #                    test_curve_x,
    #                    test_curve_y,
    #                    test_saddle])


#    test_for_nans_infs()